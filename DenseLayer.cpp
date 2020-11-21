#include"LayerCore.cpp"
#include<random>


template<size_t O>
class VecOutLayer : public virtual Layer
{
public:
	VecOutLayer(){}

	VecOutLayer(vec<O> v)
	{
		output = v;
	}
	vec<O> output;
	vec<O> ouputgrad;

};

template<size_t I , size_t O>
class DenseLayer : public VecOutLayer<O>
{
protected:
	VecOutLayer<I>* prevVec;

	Matrix<O, I> W;
	vec<O> b;

	vec<O> Z;

	Matrix<O, I> dW;
	vec<O> db;
	vec<O> dZ;


	double learningRate;

public:

	DenseLayer(VecOutLayer<I>& input, double alpha = 0.1)
	{
		prev = &input;
		input.next = this;
		prevVec = &input;
		learningRate = alpha;
	}

	virtual void preprocessForward()
	{
		Z = dot(W, prevVec->output) + b;
	}
	virtual void preprocessBackward()
	{
		dW = outer(dZ, prevVec->output);
		db = dZ;
		prevVec->outputgrad = dot(W.T(), dZ);
	}

	virtual void predict()
	{
		preprocessForward();
		for (int i = 0; i < O; i++)
			output[i] = activation(Z[i]);
	}

	virtual void backProp()
	{
		for (int i = 0; i < O; i++)
			dZ[i] = outputgrad[i] * activationgrad(Z[i]);
		
		preprocessBackward();

	}

	virtual void update()
	{
		W = W - learningRate * dW;
		b = b - learningRate * db;
	}

	virtual void initialize()
	{
		std::normal_distribution n(0, 1 / I);
		for (int i = 0; i < O; i++)
			for (int j = 0; j < I; j++)
				W[i][j] = n();

		b = 0;
		Z = 0;
		dW = 0;
		db = 0;
		dZ = 0;

	}

	virtual void attach(VecOutLayer<I> & L)
	{
		prev = &L;
		prevVec = &L
		L.next = this;
	}

};