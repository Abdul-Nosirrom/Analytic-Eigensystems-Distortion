

#include <igl/polar_dec.h>

#include "DeformationGradient.h"
#include "../Utilities/MathUtils.h"

static int count = 0;

DeformationGradient::DeformationGradient(Tet _rest)
{
	X = TranslateToOrigin(_rest).transpose(); 

	Matrix3x3 Dm;

	Dm.col(0) = X.col(1) - X.col(0);
	Dm.col(1) = X.col(2) - X.col(0);
	Dm.col(2) = X.col(3) - X.col(0);

	bool isInvertible;

	double det;

	Dm.computeInverseAndDetWithCheck(Dminv, det, isInvertible);
	
}

/// <summary>
/// The rest position is already saved upon initialization, so we pass the new deformed points
/// here and compute the deformation gradient.
/// Ensure that one of the vertices is at the origin - so we translate one point to the origin
/// for both sets of points, then compute F to remove any global translations
/// </summary>
/// <param name="_deform"></param>
void DeformationGradient::GenerateGradient(Tet _deform)
{	
	Matrix3x3 Ds;
	TetR deform = TranslateToOrigin(_deform).transpose();

	Ds.col(0) = deform.col(1) - deform.col(0);
	Ds.col(1) = deform.col(2) - deform.col(0);
	Ds.col(2) = deform.col(3) - deform.col(0);

	F = Ds * Dminv;

	CalculateInvariants();
	CalculateGradient();

	f = Vectorize(F);

	CalculateModifiedSVDPolar();
}

void DeformationGradient::CalculateModifiedSVDPolar()
{
	Eigen::JacobiSVD<Matrix3x3> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
	svd.computeU(); svd.computeV();
	U = svd.matrixU();
	V = svd.matrixV();
	SigmaVec = svd.singularValues();

	Sigma.setIdentity();
	Sigma(0, 0) = SigmaVec(0); Sigma(1, 1) = SigmaVec(1); Sigma(2, 2) = SigmaVec(2);

	Matrix3x3 I; I.setIdentity();
	auto prod = U * V.transpose();
	I(2, 2) = prod.determinant();

	double detU = U.determinant();
	double detV = V.determinant();

	if (detU < 0 && detV > 0) U *= I;
	else if (detU > 0 && detV < 0) V *= I;

	Sigma = Sigma * I;

	R = U * V.transpose();
	S = V * Sigma * V.transpose();
}

void DeformationGradient::CalculateInvariants()
{
	I1 = S.trace();
	I2 = S.squaredNorm();
	I3 = S.determinant();
}

// Construct the d^2 x (3t) matrix
// Total Gradient is d^2 x (3n) matrix
// Each 3t represents a block in the 3n matrix
// Place each one in iterations of 3t
// For example rows 0-11 correspond to df0/dx0
// rows 12-23 correspond to df1/dx1 and so on
// Here we're just calculating each dfi/dxi entry
void DeformationGradient::CalculateGradient()
{
	const double m = Dminv(0, 0);
	const double n = Dminv(0, 1);
	const double o = Dminv(0, 2);
	const double p = Dminv(1, 0);
	const double q = Dminv(1, 1);
	const double r = Dminv(1, 2);
	const double s = Dminv(2, 0);
	const double t = Dminv(2, 1);
	const double u = Dminv(2, 2);

	const double t1 = -m - p - s;
	const double t2 = -n - q - t;
	const double t3 = -o - r - u;

	PFPx.setZero();
	PFPx(0, 0) = t1;
	PFPx(0, 3) = m;
	PFPx(0, 6) = p;
	PFPx(0, 9) = s;
	PFPx(1, 1) = t1;
	PFPx(1, 4) = m;
	PFPx(1, 7) = p;
	PFPx(1, 10) = s;
	PFPx(2, 2) = t1;
	PFPx(2, 5) = m;
	PFPx(2, 8) = p;
	PFPx(2, 11) = s;
	PFPx(3, 0) = t2;
	PFPx(3, 3) = n;
	PFPx(3, 6) = q;
	PFPx(3, 9) = t;
	PFPx(4, 1) = t2;
	PFPx(4, 4) = n;
	PFPx(4, 7) = q;
	PFPx(4, 10) = t;
	PFPx(5, 2) = t2;
	PFPx(5, 5) = n;
	PFPx(5, 8) = q;
	PFPx(5, 11) = t;
	PFPx(6, 0) = t3;
	PFPx(6, 3) = o;
	PFPx(6, 6) = r;
	PFPx(6, 9) = u;
	PFPx(7, 1) = t3;
	PFPx(7, 4) = o;
	PFPx(7, 7) = r;
	PFPx(7, 10) = u;
	PFPx(8, 2) = t3;
	PFPx(8, 5) = o;
	PFPx(8, 8) = r;
	PFPx(8, 11) = u;
}

/// <summary>
/// Derivatives of invariants with respect to the deformation gradient (vectorized)
/// </summary>
/// <param name="i"></param>
/// <returns></returns>
Eigen::VectorXd DeformationGradient::InvariantDerivatives(int i)
{
	Matrix3x3 sig(3, 3);

	switch (i)
	{
	case 1: return Vectorize(R);
	case 2: return 2 * f;
	case 3:
		sig(0, 0) = Sigma(0, 0) * Sigma(2, 2);
		sig(1, 1) = Sigma(2, 2) * Sigma(0, 0);
		sig(2, 2) = Sigma(0, 0) * Sigma(1, 1);
		return Vectorize(U * sig * V.transpose());
	default:
		std::cout << "Passed in wrong index woops: " + i << std::endl;
		return Vectorize(Eigen::Matrix3d::Zero());
	}
}

#pragma region TWIST_FLIP_SCALING

void DeformationGradient::SetUpTDLMatrices()
{
	Matrix3x3 I(3, 3);
	I.setIdentity();
	double sqrt2inv = 1 / pow(2, 0.5);

	// ==== Set Up D first === //
	// For D1
	I(1, 1) = 0; I(2, 2) = 0;
	D1 = U * I * V.transpose();
	I.setIdentity();
	// For D2
	I(0, 0) = 0; I(2, 2) = 0;
	D2 = U * I * V.transpose();
	I.setIdentity();
	// For D3
	I(0, 0) = 0; I(1, 1) = 0;
	D3 = U * I * V.transpose();
	I.setIdentity();

	// === Set Up T === // 
	// For T1 x-axis
	I.setZero();
	I(2, 1) = 1; I(1, 2) = -1;
	T1 = sqrt2inv * U * I * V.transpose();
	// For T2 y-axis
	I.setZero();
	I(0, 2) = -1; I(2, 0) = 1;
	T2 = sqrt2inv * U * I * V.transpose();
	// For T3 z-axs
	I.setZero();
	I(1, 0) = 1; I(0, 1) = -1;
	T3 = sqrt2inv * U * I * V.transpose();


	// === Set Up L === // 
	// For L1
	I.setZero();
	I(2, 1) = 1; I(1, 2) = 1;
	L1 = sqrt2inv * U * I * V.transpose();
	// For L2
	I.setZero();
	I(0, 2) = 1; I(2, 0) = 1;
	L2 = sqrt2inv * U * I * V.transpose();
	// For L3
	I.setZero();
	I(1, 0) = 1; I(0, 1) = 1;
	L3 = sqrt2inv * U * I * V.transpose();
	
}

#pragma endregion TWIST_FLIP_SCALING

#pragma region ENERGY_GRADIENTS_PER_QUADRATURE
/// <summary>
/// Returns a vector of size d^2 
/// </summary>
/// <returns></returns>
Eigen::VectorXd DeformationGradient::ARAPGrad()
{
	Eigen::Matrix<double, 9, 1> PpsiPFq;
	PpsiPFq.setZero();

	PpsiPFq += -2 * InvariantDerivatives(1) + InvariantDerivatives(2) + 0 * InvariantDerivatives(3);

	return PpsiPFq;
}


Eigen::VectorXd DeformationGradient::SDGrad()
{
	Eigen::Matrix<double, 9, 1> PpsiPFq;
	PpsiPFq.setZero();

	double dI1 = I1 * (I1 * I1 - I2) / (2 * I3 * I3) - 1 / I3;
	double dI2 = (-I1 * I1 + I2 + 2 * I3 * I3) / (4 * I3 * I3);
	double dI3 = -pow(I1 * I1 - I2, 2) / (4 * pow(I3, 3)) + I1 / (I3 * I3);

	PpsiPFq += dI1 * InvariantDerivatives(1) + dI2 * InvariantDerivatives(2) + dI3 * InvariantDerivatives(3);

	return PpsiPFq;
}

#pragma endregion ENERGY_GRADIENTS_PER_QUADRATURE
