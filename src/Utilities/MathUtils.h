
#include <igl/igl_inline.h>
#include <igl/barycentric_coordinates.h>

using Tet = Eigen::Matrix<double, 4, 3>;

static inline int Delta(int i, int j) { return i == j; }

static inline Eigen::VectorXd Vectorize(Eigen::MatrixXd F)
{
	Eigen::VectorXd f(9);

	int index = 0;

	for (int i = 0; i < F.rows(); i++)
	{
		for (int j = 0; j < F.cols(); j++, index++)
		{
			f(index) = F(i, j);
		}
	}

	return f;
}

static inline Tet TranslateToOrigin(Tet tet)
{
	Tet translatedTet = tet;

	for (int i = 0; i < translatedTet.rows(); i++)
	{
		for (int j = 0; j < translatedTet.cols(); j++)
		{
			translatedTet(i, j) -= tet(0, j);
		}
	}

	return translatedTet;
}


static inline std::vector<Tet> GetTetVertices(Eigen::MatrixXd &V, Eigen::MatrixXi &T)
{
	std::vector<Tet> tets;

	for (int i = 0; i < T.rows(); i++)
	{
		Eigen::MatrixXd tet(4, 3);
		
		int a = T(i, 0), b = T(i, 1), c = T(i, 2), d = T(i, 3);

		tet <<	V(a, 0), V(a, 1), V(a, 2),
				V(b, 0), V(b, 1), V(b, 2),
				V(c, 0), V(c, 1), V(c, 2),
				V(d, 0), V(d, 1), V(d, 2);


		tets.push_back(tet);
	}

	return tets;

}