

#include "cuClusterLinearity.h"

#define CLUSTER_LINEARITY_DEBUG

namespace TNet
{

// TROY:: Propagate is the same as <biasedlinearity> with the combined weights
void CuClusterLinearity::PropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
{
	//Y.SetConst(0.0);
	Y.AddScaledRow(1.0, mBias, 0.0);
	Y.Gemm('N', 'N', 1.0, X, mLinearity, 1.0);
}

// TROY:: Backpropagation is also the same as <biasedlinearity> with the combined weights
void CuClusterLinearity::BackpropagateFnc(const CuMatrix<BaseFloat>& X, CuMatrix<BaseFloat>& Y)
{
	//Y.SetConst(0.0);
	Y.Gemm('N', 'T', 1.0, X, mLinearity, 0.0);
}

// TROY:: only the update is different
void CuClusterLinearity::Update()
{
#if 0
	//former implementation
	BaseFloat N = static_cast<BaseFloat>(GetInput().Rows());

	mLinearityCorrection.Gemm('T','N',-mLearningRate/N,GetInput(),GetErrorInput(),mMomentum);
	mBiasCorrection.AddColSum(-mLearningRate/N,GetErrorInput(),mMomentum);

	//regularization weight decay
	mLinearityCorrection.AddScaled(-mLearningRate*mWeightcost,mLinearity,1.0);

	mLinearity.AddScaled(1.0,mLinearityCorrection,1.0);
	mBias.AddScaled(1.0,mBiasCorrection,1.0);
#endif

#if 1
	//new implementation
	BaseFloat N = 1;
	if (mGradDivFrm)
	{
		N = static_cast<BaseFloat>(GetInput().Rows());
	}

	BaseFloat mmt_gain = static_cast<BaseFloat>(1.0 / (1.0 - mMomentum));

	N *= mmt_gain;

	// Prepare the input and errors for use


	// compute the gradients and update corresponding xforms and the combined weights
	for (int cid =0; cid <mNInstances; ++cid)
	{
		// compute the gradients


		// update the xform


		// update the corresponding combined weights


	}

	mLinearityCorrection.Gemm('T', 'N', 1.0, GetInput(), GetErrorInput(), mMomentum);
	mBiasCorrection.AddColSum(1.0, GetErrorInput(), mMomentum);

	mLinearity.AddScaled(-mLearningRate / N, mLinearityCorrection, 1.0);
	mBias.AddScaled(-mLearningRate / N, mBiasCorrection, 1.0);

	//regularization weight decay (from actual weights only)
	BaseFloat L2_decay = -mLearningRate * mWeightcost * (mGradDivFrm ? 1.0 : GetInput().Rows());
	mLinearity.AddScaled(L2_decay, mLinearity, 1.0);
#endif

	// update the cluster xforms
	BaseFloat N = 1;
	if (mGradDivFrm)
	{
		N = static_cast<BaseFloat>(GetInput().Rows());
	}

	BaseFloat mmt_gain = static_cast<BaseFloat>(1.0 / (1.0 - mMomentum));

	N *= mmt_gain;

}

void CuClusterLinearity::ReadFromStream(std::istream& rIn)
{
	//number of instances of shared weights in layer
	rIn >> std::ws >> mNInstances;
	if (mNInstances < 1)
	{
		std::ostringstream os;
		os << "Bad number of instances:" << mNInstances;
		Error(os.str());
	}

#ifdef CLUSTER_LINEARITY_DEBUG
	printf("===== cuClusterLinearity Information =====\n");
	printf("Number of clusters: %d\n", mNInstances);
#endif

	// load cluster xforms
	for (int cid = 0; cid < mNInstances; ++cid)
	{
#ifdef CLUSTER_LINEARITY_DEBUG
		printf("#cluster %d:\n", cid);
#endif
		std::string ss;
		int numclasses;
		rIn >> std::ws >> ss >> numclasses;
		if (ss.compare("c") != 0 || numclasses < 1)
		{
			std::ostringstream os;
			os << "Bad format of cluster linear xform instances:" << ss << " " << numclasses;
			Error(os.str());
		}
#ifdef CLUSTER_LINEARITY_DEBUG
		printf("\tnumclasses: %d [", numclasses);
#endif
		std::vector<int> labids;
		int val;
		for (int lid = 0; lid < numclasses; ++lid)
		{
			rIn >> val;
			if (val < 0 || val >= GetNOutputs())
			{
				std::ostringstream os;
				os << "Invalid class id in cluster:" << val;
				Error(os.str());
			}
			labids.push_back(val);
#ifdef CLUSTER_LINEARITY_DEBUG
			printf("%d ", val);
#endif
		}
		mClusterMap.push_back(labids);
#ifdef CLUSTER_LINEARITY_DEBUG
		printf("]\n");
#endif

		// read in transform
		//matrix is stored transposed as SNet does
		BfMatrix transpose;
		rIn >> transpose;
		mClusterLinearity_host.Copy(BfMatrix(transpose, TRANS));

		//biases stored normally
		BfVector bias;
		rIn >> bias;
		mClusterBias_host.Copy(bias);

		if (transpose.Cols() * transpose.Rows() == 0)
		{
			Error("Missing linearity matrix in network file");
		}
		if (bias.Dim() == 0)
		{
			Error("Missing bias vector in network file");
		}
		if (mClusterLinearity_host.Cols() != GetNInputs() || mClusterLinearity_host.Rows() != GetNInputs() || mClusterBias_host.Dim() != GetNInputs())
		{
			std::ostringstream os;
			os << "Wrong dimensionalities of matrix/vector in network file\n" << "Inputs:" << GetNInputs() << "Outputs:" << GetNInputs() << "\n" << "linearityCols:" << mClusterLinearity_host.Cols() << "linearityRows:" << mClusterLinearity_host.Rows() << "biasDims:" << mClusterBias_host.Dim() << "\n";
			Error(os.str());
		}

		std::string bfname(mpTempBasisDir);
		bfname += "/clusterxform";
		char buf[33];
		sprintf(buf, "%d", cid);
		bfname.append(buf);

		std::ofstream fbasis(bfname.c_str());
		if (!fbasis.good())
		{
			Error(std::string("Error, cannot write cluster xform: ") + bfname);
		}
		fbasis << transpose;
		fbasis << bias;
		fbasis.close();

#ifdef CLUSTER_LINEARITY_DEBUG
		printf("\tTransform Dim: %d x %d; Bias Dim: %d\n", mClusterLinearity_host.Cols(), mClusterLinearity_host.Rows(), mClusterBias_host.Dim());
#endif

	}

	// constant original weights
	//matrix is stored transposed as SNet does
	BfMatrix transpose;
	rIn >> transpose;
	mConstLinearity_host.Copy(BfMatrix(transpose, TRANS));
	BfVector bias;
	rIn >> bias;
	mConstBias_host.Copy(bias);

	if (transpose.Cols() * transpose.Rows() == 0)
	{
		Error("Missing linearity matrix in network file");
	}
	if (bias.Dim() == 0)
	{
		Error("Missing bias vector in network file");
	}
	if (mConstLinearity_host.Cols() != GetNOutputs() || mConstLinearity_host.Rows() != GetNInputs() || mConstBias_host.Dim() != GetNOutputs())
	{
		std::ostringstream os;
		os << "Wrong dimensionalities of matrix/vector in network file\n" << "Inputs:" << GetNInputs() << "Outputs:" << GetNOutputs() << "\n" << "linearityCols:" << mConstLinearity_host.Cols() << "linearityRows:" << mConstLinearity_host.Rows() << "biasDims:" << mConstBias_host.Dim() << "\n";
		Error(os.str());
	}

#ifdef CLUSTER_LINEARITY_DEBUG
	printf("Const Weight Dim: %d x %d; Bias Dim: %d\n", mConstLinearity_host.Cols(), mConstLinearity_host.Rows(), mConstBias_host.Dim());
#endif

	// combined weights
	rIn >> transpose;
	mLinearity_host.Copy(BfMatrix(transpose, TRANS));
	mLinearity.CopyFrom(mLinearity_host);
	//biases stored normally
	rIn >> bias;
	mBias_host.Copy(bias);
	mBias.CopyFrom(bias);

	if (transpose.Cols() * transpose.Rows() == 0)
	{
		Error("Missing linearity matrix in network file");
	}
	if (bias.Dim() == 0)
	{
		Error("Missing bias vector in network file");
	}
	if (mLinearity.Cols() != GetNOutputs() || mLinearity.Rows() != GetNInputs() || mBias.Dim() != GetNOutputs())
	{
		std::ostringstream os;
		os << "Wrong dimensionalities of matrix/vector in network file\n" << "Inputs:" << GetNInputs() << "Outputs:" << GetNOutputs() << "\n" << "linearityCols:" << mLinearity.Cols() << "linearityRows:" << mLinearity.Rows() << "biasDims:" << mBias.Dim() << "\n";
		Error(os.str());
	}

#ifdef CLUSTER_LINEARITY_DEBUG
	printf("Combined Weight Dim: %d x %d; Bias dim: %d\n", mLinearity.Cols(), mLinearity.Rows(), mBias.Dim());
	printf("===== cuClusterLinearity Information Done =====\n");
#endif
}

void CuClusterLinearity::WriteToStream(std::ostream& rOut)
{
	// number of clusters
	rOut << mNInstances << std::endl;

	BfMatrix transpose;
	BfVector bias;

	// the transforms are all square matrices
	for (int cid = 0; cid < mNInstances; ++cid)
	{
		rOut << "c " << mClusterMap[cid].size();
		for (int lid = 0; lid < mClusterMap[cid].size(); ++lid)
			rOut << " " << mClusterMap[cid][lid];
		rOut << std::endl;

		// load the xform from file
		std::string bfname(mpTempBasisDir);
		bfname += "/clusterxform";
		char buf[33];
		sprintf(buf, "%d", cid);
		bfname.append(buf);

		std::ifstream fbasis(bfname.c_str());
		if (!fbasis.good())
		{
			Error(std::string("Error, cannot write cluster xform: ") + bfname);
		}
		fbasis >> transpose;
		fbasis >> bias;
		fbasis.close();

		if (transpose.Cols() * transpose.Rows() == 0)
		{
			Error("Missing linearity matrix in network file");
		}
		if (bias.Dim() == 0)
		{
			Error("Missing bias vector in network file");
		}
		if (transpose.Cols() != GetNInputs() || transpose.Rows() != GetNInputs() || bias.Dim() != GetNInputs())
		{
			std::ostringstream os;
			os << "Wrong dimensionalities of matrix/vector in network file\n" << "Inputs:" << GetNInputs() << "Outputs:" << GetNInputs() << "\n" << "linearityCols:" << transpose.Cols() << "linearityRows:" << transpose.Rows() << "biasDims:" << bias.Dim() << "\n";
			Error(os.str());
		}

		rOut << transpose;
		rOut << bias;

	}

	// constant weights
	rOut << BfMatrix(mConstLinearity_host, TRANS);
	rOut << mConstBias_host;

	// combined weights
	//matrix is stored transposed as SNet does
	mLinearity.CopyTo(transpose);
	rOut << BfMatrix(transpose, TRANS);
	//biases stored normally
	mBias.CopyTo(bias);
	rOut << bias;
	rOut << std::endl;
}

} //namespace

