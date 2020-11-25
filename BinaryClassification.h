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

	double forwardprop(vec<I>& in, double label)
	{
		Layer* iter = inputLayer;
		inputLayer->set(in);
		while(iter != outputLayer)
		{
			iter->forwardprop();
			iter = iter->next;
		}
		outputLayer->label = label;
		outputLayer->forwardprop();

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

template<size_t I , size_t N , size_t TSS>
void train(BinaryClassificationNetwork<I,N> & bcn,Matrix<TSS, I>& trainx, vec<TSS>& trainy, int numEpochs = 10)
	{
		for (int i = 0; i < numEpochs; i++)
		{
			for (int j = 0; j < TSS; j++)
			{
				bcn.forwardprop(trainx[j], trainy[j]);
				bcn.backprop();
				bcn.update();
			}

		}
	}
#endif BCNETWORK