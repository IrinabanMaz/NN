#ifndef ACTIVATIONS
#define ACTIVATIONS
#include"LayerCore.h"

/*
class Identity : public virtual Layer
{
	double activation(double x)
	{
		return x;
	}
	double activationgrad(double x)
	{
		return 1;
	}
};

*/
//ReLU activation.
class ReLU : public virtual Layer
{
	double activation(double x)
	{
		return fmax(0, x);
	}
	double activationgrad(double x)
	{
		return x > 0 ? 1 : 0;
	}
};

class LeakyReLU : public virtual Layer
{
	double activation(double x)
	{
		return fmax(0.01*x, x);
	}
	double activationgrad(double x)
	{
		return (x > 0) ? 1 : 0.01;
	}
};

class Tanh : public virtual Layer
{
	double activation(double x)
	{
		return tanh(x);
	}
	double ativationgrad(double x)
	{
		return 1 / (cosh(x) * cosh(x));
	}
};

inline double sigmoid(double x)
{
	return 1.0 / (1 + exp(-x));
}

//sigmoid Activation. For debugging, call resetvalid() 
//in derived class update function to keep flags valid.
class Sigmoid : public virtual Layer
{
	bool valid = false;
	double activation(double x)
	{
		valid = true;
		return sigmoid(x);
	}
	double activationgrad(double x)
	{
		double a = sigmoid(x);
		
		return a * (1 - a);
	}

	void resetvalid()
	{
		valid = false;
	}
};

class SigmoidCE : public virtual Layer
{

public:
	bool valid = false;
	double label;

	double activation(double x)
	{
		valid = true;
		if (label > 0.5)
			return -log(sigmoid(x));
		else 
			return -log(1 - sigmoid(x));
		
	}
	double activationgrad(double x)
	{
		double a = sigmoid(x);
		if (label > 0.5)
			return a-1;
		else
			return a;
	}



	void resetvalid()
	{
		valid = false;
	}
};

#endif ACTIVATIONS