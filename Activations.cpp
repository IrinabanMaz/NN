#include"LayerCore.cpp"



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

double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
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
	double label;
	bool valid = false;
	double activation(double x)
	{
		valid = true;
		if (label == 1.0)
			return -log(sigmoid(x));
		else if (label == 0.0)
			return -log(1 - sigmoid(x));
		else
			return 0;
	}
	double activationgrad(double x, int i)
	{
		double a = sigmoid(x);
		if (label == 1.0)
			return a - 1;
		else if (label == 0.0)
			return a;
		else
			return 0;
	}



	void resetvalid()
	{
		valid = false;
	}
};