#ifndef DENSELAYER
#define DENSELAYER
#include"LayerCore.h"
#include"Matrix.h"
#include<random>


template<size_t O>
class VecOutLayer : public virtual Layer
{
public:
	VecOutLayer() { outputgrad = 1.0; }

	VecOutLayer(vec<O> v)
	{
		output = v;
	}
	vec<O> output;
	vec<O> outputgrad;

	void set(vec<O> v)
	{
		output = v;
	}

};

template<size_t I , size_t O>
class DenseLayer : public virtual VecOutLayer<O>
{
protected:

	std::default_random_engine gen;

	VecOutLayer<I>* prevVec;

	Matrix<O, I> W;
	vec<O> b;

	vec<O> Z;

	Matrix<O, I> dW;
	vec<O> db;
	vec<O> dZ;


	double learningRate;

public:

	DenseLayer()
	{
		learningRate = 0.1;
	}

	DenseLayer(VecOutLayer<I>& input, double alpha = 0.1)
	{
		this->prev = &input;
		input.next = this;
		this->prevVec = &input;
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
			this->output[i] = this->activation(Z[i]);
	}

	virtual void backProp()
	{
		for (int i = 0; i < O; i++)
			dZ[i] = this->outputgrad[i] * this->activationgrad(Z[i]);
		
		preprocessBackward();

	}

	virtual void update()
	{
		W = W - dW * learningRate;
		b = b - db * learningRate;
	}

	virtual void initialize()
	{
		std::normal_distribution<double> n(0, 1.0 / I);
		for (int i = 0; i < O; i++)
			for (int j = 0; j < I; j++)
				W[i][j] = n(gen);

		b = 0.0;
		Z = 0.0;
		dW = 0.0;
		db = 0.0;
		dZ = 0.0;
		learningRate = 0.1;

	}

	virtual void attach(VecOutLayer<I> & L)
	{
		this->prev = &L;
		this->prevVec = &L;
		L.next = this;
	}

};




#endif DENSELAYER