#ifndef BCNETWORK
#define BCNETWORK
#include "BatchDenseLayer.h"
#include "Activations.h"

template<size_t B ,size_t I>
class BBCInputLayer : public BatchVecOutLayer<B , I>
{};

template<size_t B ,size_t N>
class BBCOutputLayer :public BatchDenseLayer<B, N, 1>, public SigmoidCE
{};

template<size_t B ,size_t I, size_t N>
class BatchBinaryClassificationNetwork
{
public:
	BBCInputLayer<B ,I>* inputLayer;
	BBCOutputLayer<B ,N>* outputLayer;


	BatchBinaryClassificationNetwork(BBCInputLayer<B , I>* in, BBCOutputLayer<B ,N>* out)
	{
		inputLayer = in;
		outputLayer = out;
	}

	void initialize()
	{
		Layer* iter = inputLayer;
		while (iter != nullptr)
		{
			iter->initialize();
			iter = iter->next;

		}
	}

	void forwardProp(Matrix<B ,I>& in, vec<B> label)
	{
		Layer* iter = inputLayer;
		inputLayer->set(in);
		while (iter != outputLayer)
		{
			iter->forwardProp();
			iter = iter->next;
		}
		
		outputLayer->preprocessForward();

		for (int b = 0; b < B; b++)
		{
			outputLayer->label = label[b];
			outputLayer->output[b][0] = outputLayer->activation(outputLayer->Z[b][0]);
		}


	}

	void backProp(vec<B> label)
	{
		Layer* iter = outputLayer;

		for (int b = 0; b < B; b++)
		{
			outputLayer->label = label[b];
			outputLayer->dZ[b][0] = outputLayer->outputgrad[b][0] * outputLayer->activationgrad(outputLayer->Z[b][0]);
		}
		outputLayer->preprocessBackward();
		iter = iter->prev;

		while (iter != inputLayer)
		{
			iter->backProp();
			iter = iter->prev;
		}

	}

	void update()
	{
		Layer* iter = inputLayer;
		while (iter != nullptr)
		{
			iter->update();
			iter = iter->next;

		}
	}

	double predict(vec<I>& in)
	{
		Layer* iter = inputLayer;
		inputLayer->set(in);
		while (iter != outputLayer)
		{
			iter->predict();
			iter = iter->next;
		}
		outputLayer->label = 1.0;
		outputLayer->predict();
		double temp = toDouble(outputLayer->predictoutput);

		return exp(-temp);
	}

};

template<size_t TSS, size_t BS, size_t I, size_t N >
void train(BatchBinaryClassificationNetwork<BS , I, N>& bcn, Matrix<TSS, I>& trainx, vec<TSS>& trainy, int numEpochs = 10)
{

	//variables for shuffling.
	std::default_random_engine gen;
	std::uniform_int_distribution<int>* r;
	vec<I> tempx;
	double tempy;
	int swapindex;

	Matrix<BS, I> batchX;
	vec<BS> batchY;

	

	for (int i = 0; i < numEpochs; i++)
	{
		//shuffle training data.
		for (int k = 0; k < TSS; k++)
		{

			r = new std::uniform_int_distribution<int>(k, TSS - 1);
			swapindex = (*r)(gen);
			tempx = trainx[k];
			tempy = trainy[k];
			trainx[k] = trainx[swapindex];
			trainy[k] = trainy[swapindex];
			trainx[swapindex] = tempx;
			trainy[swapindex] = tempy;
			delete r;

		}

		//loop batch gradient descent through training data.
		for (int j = 0; j < TSS / BS; j++)
		{
			batchX = submatrix<BS, I>(trainx, j * BS, 0);
			batchY = subvector<BS>(trainy, j * BS);
			bcn.forwardProp(batchX, batchY);
			bcn.backProp(batchY);
			bcn.update();
		}

	}
}
#endif BCNETWORK
