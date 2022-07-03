
#include "Simulator.h"
#include "DeformationGradient.h"
#include <igl/line_search.h>
#include "../Utilities/MathUtils.h"

#pragma region STATIC_MEMBER_INITIALIZATION
std::vector<DeformationGradient> Simulator::FGradContainer{};
MeshData _currentMesh = MeshData(), _mesh1 = MeshData(), _mesh2 = MeshData();
#pragma endregion STATIC_MEMBER_INITIALIZATION

#pragma region SETUP_SIMULATOR

Simulator::Simulator()
{
	energy = ARAPEnergy;
	energyGradient = ARAPGradient;
	energyEigensystem = ARAPEigensystem;
}


/// <summary>
/// Initialize Rest Positions and save them, as well as begin initializing deformation gradient
/// Which means creating its instances and pushing it into a list, as well as computing the inverses
/// of Dm as they do not change throughout.
/// </summary>
/// <param name="verts"></param>
/// <param name="faces"></param>
void Simulator::SetUpMeshes(MeshData mesh)
{
	std::vector<Tet> tets(mesh.T.rows());
	tets = GetTetVertices(mesh.V, mesh.T);

	// Initialize deformation gradient - actual gradient hasn't been computed yet, computed in the solver below
	for (auto tet : tets)
	{
		DeformationGradient F(tet);
		FGradContainer.push_back(F);
	}
}

void Simulator::InputDeformedMesh(MeshData mesh)
{
	_currentMesh = mesh;
	RegenerateQuantities();
}

void Simulator::RegenerateQuantities()
{
	std::vector<Tet> tets(_currentMesh.T.rows());
	tets = GetTetVertices(_currentMesh.V, _currentMesh.T);

	int i = 0;
	for (auto tet : tets)
	{
		FGradContainer[i].GenerateGradient(tet);
		i++;
	}
}

void Simulator::ChangeEnergy(int energyVal)
{
	switch (energyVal)
	{
	case 0: //ARAP
		energy = ARAPEnergy;
		energyGradient = ARAPGradient;
		energyEigensystem = ARAPEigensystem;
		break;
	case 1: // SD
		energy = SDEnergy;
		energyGradient = SDGradient;
		energyEigensystem = SDEigensystem;
		break;
	default:
		std::cout << "Woops, how did this value get here" << std::endl;
		break;
	}
}
#pragma endregion SETUP_SIMULATOR

#pragma region SOLVER
/// <summary>
/// Set Up Global PFPX from each PFPX quadrature block 
/// Since we sum up each segment, it'll be very sparse where the only filled portion of the matrix is
/// entries in rows (12q) -> (12q + 12)
/// </summary>
Eigen::MatrixXd Simulator::SetUpGlobalPFPX(int q)
{
	Eigen::MatrixXd PFPX(_currentMesh.V.rows() , 9);

	int row0 = 12 * q;
	PFPX.setZero(); 
	// Setup global F gradient blocks
	for (int i = 0; i < 12; i++)
	{
		PFPX.row(row0 + i) = FGradContainer[q].PFPx.row(i);
	}

	return PFPX;
}

/// <summary>
/// The actual step in Algorithm 1 as described in the paper
/// Fix your constraints!!!!
/// </summary>
/// <param name="n"></param>
void Simulator::Projected_Newton_Solver()
{
	Eigen::MatrixXd b, d; 
	Eigen::MatrixXd Hi;	  
	double step_size = 0.5f;
	do
	{
		b = energyGradient(*this);
		Hi = Project_Hessian(); 

		d = -Hi.inverse() * b;
		double a = igl::line_search(Hi, d, step_size, energy);
		Clean_Constraints(d);

		_currentMesh.V += a * d;

		// Recalculate per quadrature quantities for newly iterated mesh
		RegenerateQuantities();

	} while (b.norm() < _threshold);
}

// Tried setting constraints within the deformation gradient itself so it evals to zero, didn't work out so hopefully this does
void Simulator::Clean_Constraints(Eigen::MatrixXd& d)
{
	for (auto vertID : constraintVertIDs)
	{
		for (int col = 0; col < d.cols(); col++)
		{
			d(vertID, col) = 0;
		}
	}
}

/// <summary>
/// Takes in the new Verts as input, and computes the hessian
/// </summary>
/// <param name="x"></param>
/// <returns></returns>
Eigen::MatrixXd Simulator::Project_Hessian()
{
	Eigen::MatrixXd H(_currentMesh.V.rows(), _currentMesh.V.rows());
	H.setZero();
	Eigen::VectorXd Hq(9);
	Hq.setZero();

	int i = 0;

	for (auto q : tets_rest)
	{
		// Generate Gradient call already computes SVD decomposition and necessary per quadrature quantities
		DeformationGradient F = FGradContainer[i];

		Eigen::MatrixXd PFPX = SetUpGlobalPFPX(i);		// Sparse
		double vol = igl::volume_single(q.row(0), q.row(1), q.row(2), q.row(3));
		// dPSI/dX = sum_q |q| (df_q/dX)^T * dPSI_q/df_q

		Eigensystem system = energyEigensystem(i);
		for (int j = 0; j < system.lambdas.size(); j++)
		{
			// Clamp Eigenvalues
			Hq += std::max(system.lambdas[j], 0.0) * system.es[j] * system.es[j].transpose();
		}

		// Then Update Hessian based on energy eigensystem
		H += vol * PFPX.transpose() * Hq * PFPX;
		i++;
	}

	return H;
}

#pragma endregion SOLVER

#pragma region ENERGY_EVALUATION

/// <summary>
/// Evaluate a given energy over the entire mesh
/// where X - vertex matrix not used but necessary for the igl line search call
/// </summary>
/// <param name="X"></param>
/// <returns></returns>
double Simulator::ARAPEnergy(Eigen::MatrixXd& X)
{
	//Psi(X) = Sum_{quadrature q} Psi_q(F) * |q|	
	double psi = 0;
	double vol = 1; 

	for (auto F : FGradContainer)
	{
		psi += vol * (F.F - F.R).squaredNorm(); 
	}

	return psi;
}

double Simulator::SDEnergy(Eigen::MatrixXd& X)
{
	//Psi(X) = Sum_{quadrature q} Psi_q(F) * |q|

	double psi = 0;
	double vol = 1;

	for (auto F : FGradContainer)
	{
		psi += vol * (0.5 * F.I2 + 0.125 * pow((F.I1 * F.I1 - F.I2)/2,2) - F.I1/F.I3); 
	}

	return psi;
}

#pragma endregion ENERGY_EVALUATION

/// DO THESE NOW
#pragma region ENERGY_GRADIENTS

/// <summary>
/// Construct the global gradient from per-quadrature quantities
/// </summary>
/// <param name="X"></param>
/// <returns></returns>
Eigen::MatrixXd Simulator::ARAPGradient(Simulator &sim)
{
	auto F = FGradContainer;
	Eigen::MatrixXd PpsiPx(sim._meshDef.V.rows(), 9);
	PpsiPx.setZero();
	double vol = 1;

	for (int q = 0; q < F.size(); q++)
	{
		PpsiPx += vol * sim.SetUpGlobalPFPX(q).transpose() * F[q].ARAPGrad();
	}

	return PpsiPx;
}

Eigen::MatrixXd Simulator::SDGradient(Simulator &sim)
{
	auto F = FGradContainer;
	Eigen::MatrixXd PpsiPx(sim._meshDef.V.rows(), 9);
	PpsiPx.setZero();
	double vol = 1;

	for (int q = 0; q < F.size(); q++)
	{
		PpsiPx += vol * sim.SetUpGlobalPFPX(q).transpose() * F[q].SDGrad();
	}

	return PpsiPx;
}

#pragma endregion ENERGY_GRADIENTS

#pragma region ENERGY_EIGENSYSTEMS
/// <summary>
/// t in the paper are the twist vectorized matrices (T) corresponding to x,y,z axis
/// l in the paper are the flip vectorized matrices (L) corresponding to x,y,z axis
/// THIS IS FOR ARAP, not implemented for SD yet but those are already given so just testing on ARAP for now
/// </summary>
/// <param name="energy"></param>
/// <param name="F"></param>
/// <returns></returns>
Eigensystem Simulator::ARAPEigensystem(int q)
{
	Eigensystem eigensystem;
	DeformationGradient F = FGradContainer[q];

	// Eigenvalues
	std::vector<double> lambdas;

	double sigma1 = F.Sigma(0, 0);
	double sigma2 = F.Sigma(1, 1);
	double sigma3 = F.Sigma(3, 3);

	lambdas.push_back(2 - 4 / (sigma2 + sigma3));
	lambdas.push_back(2 - 4 / (sigma3 + sigma1));
	lambdas.push_back(2 - 4 / (sigma1 + sigma2));
	for (int i = 4; i <= 9; i++) lambdas.push_back(2);
	

	// Eigenvectors
	std::vector<Eigen::VectorXd> eigenVectors;

	eigenVectors.push_back(Vectorize(F.T1));
	eigenVectors.push_back(Vectorize(F.T2));
	eigenVectors.push_back(Vectorize(F.T3));

	eigenVectors.push_back(Vectorize(F.L1));
	eigenVectors.push_back(Vectorize(F.L2));
	eigenVectors.push_back(Vectorize(F.L3));

	eigenVectors.push_back(Vectorize(F.D1));
	eigenVectors.push_back(Vectorize(F.D2));
	eigenVectors.push_back(Vectorize(F.D3));

	// store them
	eigensystem.lambdas = lambdas;
	eigensystem.es = eigenVectors;

	return eigensystem;
}

/// <summary>
/// Finish this too
/// </summary>
/// <param name="q"></param>
/// <returns></returns>
Eigensystem Simulator::SDEigensystem(int q)
{
	return Eigensystem();
}

#pragma endregion ENERGY_EIGENSYSTEMS