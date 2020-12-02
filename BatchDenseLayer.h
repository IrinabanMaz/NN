#ifndef DENSELAYER
#define DENSELAYER
#include"LayerCore.h"
#include"Matrix.h"
#include<random>


template<size_t B , size_t O>
class BatchVecOutLayer : public virtual Layer
{
public:
	BatchVecOutLayer() { outputgrad = 1.0; }

	BatchVecOutLayer(Matrix<B,O> D)
	{
		output = D;
	}
	Matrix<B,O> output;
	Matrix<B,O> outputgrad;
	vec<O> predictoutput;

	void set(Matrix<B,O> D)
	{
		output = D;
	}
	void set(vec<O> v)
	{
		predictoutput = v;
	}


};

template<size_t B, size_t I, size_t O>
class BatchDenseLayer : public virtual BatchVecOutLayer<B , O>
{
public:

	std::default_random_engine gen;

	BatchVecOutLayer<B , I>* prevVec;

	Matrix<O, I> W;
	vec<O> b;

	Matrix<B , O> Z;
	vec<O> predictZ;

	Matrix<O, I> dW;
	vec<O> db;
	Matrix<B,O> dZ;


	double learningRate;

public:

	BatchDenseLayer(double alpha = 0.1)
	{
		learningRate = alpha;
	}

	BatchDenseLayer(BatchVecOutLayer<B ,I>& input, double alpha = 0.1)
	{
		this->prev = &input;
		input.next = this;
		this->prevVec = &input;
		learningRate = alpha;
	}

	virtual void preprocessForward()
	{
		Z = outer(prevVec->output, W) + b;
	}
	virtual void preprocessBackward()
	{
		dW =  inner(dZ, prevVec->output) / static_cast<double>(B);
		db =   rowsum(dZ)/ static_cast<double>(B);
		prevVec->outputgrad = dot(dZ, W);
	}

	virtual void forwardProp()
	{
		preprocessForward();
		for(int ba = 0; ba < B; ba++)
			for(int i = 0; i < O; i++)
				this->output[ba][i] = this->activation(Z[ba][i]);
	}

	virtual void predict()
	{
		predictZ = dot(W, prevVec->predictoutput) + b;

		for (int i = 0; i < O; i++)
			this->predictoutput[i] = this->activation(predictZ[i]);
	}

	virtual void backProp()
	{
		for(int ba = 0; ba < B; ba++)
			for (int i = 0; i < O; i++)
				dZ[ba][i] = this->outputgrad[ba][i] * this->activationgrad(Z[ba][i]);

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

	}

	virtual void attach(BatchVecOutLayer<B , I>& L)
	{
		this->prev = &L;
		this->prevVec = &L;
		L.next = this;
	}

};

template<size_t B, size_t O>
class BatchDropoutLayer : public BatchVecOutLayer<B ,O>
{
private:
	double keepRate;
	std::default_random_engine gen;

	BatchVecOutLayer<B , O>* prevVec;

	vec<O> mask;
public:
	BatchDropoutLayer(double r)
	{
		keepRate = r;
		prevVec = nullptr;
	}

	virtual void forwardProp()
	{
		std::uniform_real_distribution<double> p(0.0, 1.0);
		for (int i = 0; i < O; i++)
			mask[i] = (p(gen) < keepRate) ? 1.0 : 0.0;

		this->output = this->prevVec->output * mask;
	}

	void backProp()
	{
		this->prevVec->outputgrad = this->outputgrad * mask;
	}

	void predict()
	{
		this->predictoutput = this->prevVec->predictoutput;
	}


	void attach(BatchVecOutLayer<B,O>& L)
	{
		this->prev = &L;
		this->prevVec = &L;
		L.next = this;
	}
};


#endif DENSELAYER