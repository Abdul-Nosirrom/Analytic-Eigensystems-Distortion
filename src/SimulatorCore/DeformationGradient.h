#include <igl/eigs.h>

/// <summary>
/// Class to handle the deformation gradient and all per-quadrature quantities associated with it
/// </summary>
class DeformationGradient
{
public:
	using Matrix3x3 = Eigen::Matrix3d;
	using Matrix2x2 = Eigen::Matrix2d; 
	using Matrix9x12 = Eigen::Matrix<double, 9, 12>;
	using Tet = Eigen::Matrix<double, 4, 3>;
	using TetR = Eigen::Matrix<double, 3, 4>;
	using Vec = Eigen::VectorXd;

	DeformationGradient(Tet _rest);

	void GenerateGradient(Tet _deform);

	Eigen::VectorXd InvariantDerivatives(int i);

	// Energy Gradients per quadrature
	Eigen::VectorXd ARAPGrad();
	Eigen::VectorXd SDGrad();

	Matrix3x3 F;
	Vec f;	// Represents the vectorized deofrmation gradient
	TetR X;	// Represents the rest positions of the quadrature

	// Singular Value Decomposition Matrices and Vector
	Eigen::Vector3d SigmaVec;
	Matrix3x3 U, V, Sigma;
	Matrix9x12 PFPx;
	double I1, I2, I3;
	
	Matrix3x3 S, R;

	// Twist, Scaling, and Flip eigenvectors, how do I get all of these?
	Matrix3x3 T1, T2, T3;
	Matrix3x3 D1, D2, D3;
	Matrix3x3 L1, L2, L3;

private:
	void CalculateInvariants();

	void CalculateGradient();

	void SetUpTDLMatrices();

	void CalculateModifiedSVDPolar();

	// Precomputed inverse
	Matrix3x3 Dminv;

	// Singular Value Decomposition of the deformation gradient
	Eigen::BDCSVD<Matrix3x3> Fsvd;

	
};