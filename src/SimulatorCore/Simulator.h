
#include <igl/polar_dec.h>
#include <igl/hessian.h>
#include <igl/hessian_energy.h>

// Forward Declaration
class DeformationGradient;

// Keep mesh data here since simulator wants to use it
struct MeshData
{
	Eigen::MatrixXd V, C; // C for colors
	Eigen::MatrixXi F;
	Eigen::MatrixXi T;
};

// Energy Eigensystem container
struct Eigensystem
{
	std::vector<double> lambdas;
	std::vector<Eigen::VectorXd> es;	// Eigenvectors, vectorized 3x3 matrices
};


/// <summary>
/// Quantities I need to compute
/// 1.) Energy Gradient - easy in terms of invariants and decomps, determines RHS
/// 2.) Projected Hessian - Used to compute displacement vectors by multiplying inverse w/ 1
/// 3.) SVD Decomposition of F, easy and done
/// 
/// General Notes
/// 1.) |q| is a volume weight associated with the quadrature point q at rest
/// 2.) Quadrature point is a tet or tri depending on input mesh
/// 3.) If tet, calculate |q| with igl::volume(V,T,Volume)
/// 4.) If tri, calculate |q| with igl::area(V,F,area)
/// 
/// Constant Values
/// 1.) Start with computing & storing values that are invariant under deformation
/// 
/// End Simulation
/// 1.) End simulation when norm gradient of energy is very small
/// 
/// </summary>

class Simulator
{
public:
	using Tet = Eigen::Matrix<double, 4, 3>;

	Simulator();

	void SetUpMeshes(MeshData mesh);

	void InputDeformedMesh(MeshData mesh);

	void ChangeEnergy(int energyVal);

	std::vector<int> constraintVertIDs;

private:

	void Projected_Newton_Solver();
	void Clean_Constraints(Eigen::MatrixXd& d);

	double _threshold = 1e-4;

	Eigen::MatrixXd Project_Hessian();
	Eigen::MatrixXd SetUpGlobalPFPX(int q);
	void RegenerateQuantities();

	// Energy Evaluation over the whole mesh
	static double ARAPEnergy(Eigen::MatrixXd& X);
	static double SDEnergy(Eigen::MatrixXd& X);

	// Energy Gradients over the whole mesh
	static Eigen::MatrixXd ARAPGradient(Simulator &sim);	// Returns Nx3 matrix
	static Eigen::MatrixXd SDGradient(Simulator &sim);

	// Energy Eigensystems for given quadrature q
	static Eigensystem ARAPEigensystem(int q);		// Gets eigenvalues/vectors of quadrature index q
	static Eigensystem SDEigensystem(int q);


	// Hold the deformation gradient for the whole mesh
	static std::vector<DeformationGradient> FGradContainer;

	MeshData _meshRest, _meshDef;

	// Hold the triangles/quadratures for both meshes
	std::vector<Tet> tets_rest, tets_def;

	// Energy Function, change depending on selection
	std::function<double(Eigen::MatrixXd&)> energy;
	std::function<Eigen::MatrixXd(Simulator&)> energyGradient;
	std::function<Eigensystem(int)> energyEigensystem;

};
