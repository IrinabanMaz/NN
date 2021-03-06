#ifndef LAYERCORE
#define LAYERCORE


#include<cmath>



class Layer
{

public:

	Layer* next;
	Layer* prev;

public:

	Layer()
	{
		next = nullptr;
		prev = nullptr;
	}

	virtual void initialize(){}

	//preprocessing functions for layer.
	virtual void preprocessForward(){}
	virtual void preprocessBackward(){}

	//activation function information for layer.
	virtual double activation(double x) { return x; }
	virtual double activationgrad(double x) { return 1; }

	//parameter update function
	virtual void update(){}

	//cost function terms for regularization.
	virtual double cost() { return prev->cost(); }
	virtual void costgrad(){}

	//prediction function.
	virtual void predict(){}
	
	//forward prop function
	virtual void forwardProp(){predict();}

	virtual void backProp(){}

};



#endif LAYERCORE