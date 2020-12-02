#ifndef BCNETWORK
#define BCNETWORK
#include "DenseLayer.h"
#include "Activations.h"

template<size_t I>
class BCInputLayer : public VecOutLayer<I>
{};

template<size_t N>
class BCOutputLayer :public DenseLayer<N, 1>, public SigmoidCE
{};

template<size_t I, size_t N>
class BinaryClassificationNetwork
{
public:
	BCInputLayer<I>* inputLayer;
	BCOutputLayer<N>* outputLayer;


	BinaryClassificationNetwork(BCInputLayer<I>* in, BCOutputLayer<N>* out)
	{
		inputLayer = in;
		outputLayer = out;
	}

	void initialize()
	{
		Layer* iter = inputLayer;
		while(iter != nullptr)
		{
			iter->initialize();
			iter = iter->next;
			
		}
	}

	double forwardProp(vec<I>& in, double label)
	{
		Layer* iter = inputLayer;
		inputLayer->set(in);
		while(iter != outputLayer)
		{
			iter->forwardProp();
			iter = iter->next;
		}
		outputLayer->label = label;
		outputLayer->forwardProp();

		return toDouble(outputLayer->output);
	}

	void backprop()
	{
		Layer* iter = outputLayer;
		while(iter != inputLayer)
		{
			iter->backProp();
			iter = iter->prev;
		}

	}

	void update()
	{
		Layer* iter = inputLayer;
		while(iter != nullptr)
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
		double temp = toDouble(outputLayer->output);

		return exp(-temp);
	}

};

template<size_t TSS , size_t I , size_t N >
void train(BinaryClassificationNetwork<I,N> & bcn,Matrix<TSS, I>& trainx, vec<TSS>& trainy, int numEpochs = 10)
	{

	//variables for shuffling.
	static std::array<int, TSS> permutation;
	static bool assigned = false;
	std::default_random_engine gen;
	std::uniform_int_distribution<int>* r;
	int temp;
	int swapindex;

	//set to identity permutation on first call.
	if (!assigned)
	{
		for (int i = 0; i < TSS; i++)
			permutation[i] = i;

		assigned = true;
	}

		for (int i = 0; i < numEpochs; i++)
		{
			//loop through permuation to shuffle.
			for (int k = 0; k < TSS; k++)
			{
				
				r = new std::uniform_int_distribution<int>(k, TSS - 1);
				swapindex = (*r)(gen);
				temp = permutation[k];
				permutation[k] = permutation[swapindex];
				permutation[swapindex] = temp;
				delete r;

			}
		    
			//loop stochastic gradient descent through training data.
			for (int j = 0; j < TSS; j++)
			{
				bcn.forwardProp(trainx[permutation[j]], trainy[permutation[j]]);
				bcn.backprop();
				bcn.update();
			}

		}
	}
#endif BCNETWORK