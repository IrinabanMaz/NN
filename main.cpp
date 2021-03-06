#include "BatchBC.h"


template<size_t B , size_t I, size_t O>
class testHiddenLayer : public BatchDenseLayer<B,I,O> ,public LeakyReLU
{
public:
	testHiddenLayer(double alpha = 0.1)
	{
		this->learningRate = alpha;
	}
};


//uses a neural network to learn the unit ball in R^n
int main()
{
	//set data sizes
	const size_t TRAINING_SET_SIZE = 1 << 14;
	const size_t BATCH_SIZE = 1 << 6;
	const size_t TEST_SET_SIZE = 100;

	//set input size.
	const size_t INPUT_SIZE = 2;

	//training data storage
	static Matrix<TRAINING_SET_SIZE, INPUT_SIZE> trainx;
	static vec<TRAINING_SET_SIZE> trainy;

	//test data storage
	static Matrix<TEST_SET_SIZE, INPUT_SIZE>  testx;
	static vec<TEST_SET_SIZE> testy;

	//set up RNG.
	std::default_random_engine gen;
	std::uniform_real_distribution<double> p(-1.0, 1.0);

	//Generate training data.
	for (int i = 0; i < TRAINING_SET_SIZE; i++)
	{
		for(int j = 0; j < INPUT_SIZE; j++)
			trainx[i][j] = p(gen);
		

		if (dot(trainx[i], trainx[i]) < 1)
			trainy[i] = 1.0;
		else
			trainy[i] = 0.0;

	}

	//generate test data.
	for (int i = 0; i < TEST_SET_SIZE; i++)
	{
		for(int j = 0; j < INPUT_SIZE;j++)
			testx[i][j] = p(gen);

		if (dot(testx[i], testx[i]) < 1)
			testy[i] = 1.0;
		else
			testy[i] = 0.0;

	}
	
	//set network structure parameters.
	const size_t HIDDEN_LAYER1_SIZE = 10;
	const size_t HIDDEN_LAYER2_SIZE = 6;
	//determine hyperpaameters
	const double KEEP_RATE = 0.5;
	const double LEARNING_RATE = 0.01;

	//declare layers
	static BBCInputLayer<BATCH_SIZE , INPUT_SIZE> Input;
	static testHiddenLayer<BATCH_SIZE ,INPUT_SIZE, HIDDEN_LAYER1_SIZE> Layer1(LEARNING_RATE);
	static BatchDropoutLayer<BATCH_SIZE, HIDDEN_LAYER1_SIZE> Dropout1(KEEP_RATE);
	static testHiddenLayer<BATCH_SIZE, HIDDEN_LAYER1_SIZE, HIDDEN_LAYER2_SIZE> Layer2(LEARNING_RATE);
	static BatchDropoutLayer<BATCH_SIZE, HIDDEN_LAYER2_SIZE> Dropout2(KEEP_RATE);
	static BBCOutputLayer<BATCH_SIZE, HIDDEN_LAYER2_SIZE> Output;

	//attach layers
	Layer1.attach(Input);
	Dropout1.attach(Layer1);
	Layer2.attach(Dropout1);
	Dropout2.attach(Layer2);
	Output.attach(Dropout2);

	//create network interface.
	BatchBinaryClassificationNetwork<BATCH_SIZE,INPUT_SIZE, HIDDEN_LAYER2_SIZE> testBC(&Input , &Output);


	testBC.initialize();
	
	double predicty;
	double correcttrainingcount = 0.0;
	double correcttestcount = 0.0;
	double acc;
	double testacc;
	

	for (int j = 1; j <= 100; j++)
	{
		//train for 100 epochs.
		train(testBC, trainx, trainy, 100);

		//measure performance on training set.
		for (int i = 0; i < TRAINING_SET_SIZE; i++)
		{
			predicty = testBC.predict(trainx[i]);
			//std::cout << predicty << std::endl;
			predicty = ((predicty > 0.5) ? 1.0 : 0.0);

			if (predicty == trainy[i])
				correcttrainingcount++;
		}
		//measure performance on test set.
		for (int i = 0; i < TEST_SET_SIZE; i++)
		{
			predicty = testBC.predict(testx[i]);
			//std::cout << predicty << std::endl;
			predicty = ((predicty > 0.5) ? 1.0 : 0.0);

			if (predicty == testy[i])
				correcttestcount++;
		}

		acc = correcttrainingcount / TRAINING_SET_SIZE;
		testacc = correcttestcount / TEST_SET_SIZE;

		std::cout << "Train set accuracy after " << j * 100 << " epochs: " << acc * 100 << "%" << std::endl;
		std::cout << "Dev set accuracy after " << j * 100  << " epochs: " << testacc * 100 << "%" << std::endl;
	

		correcttrainingcount = 0;
		correcttestcount = 0;
	}
	
	return 0;

}
























//Test for linear algbra functions.
/*
int main()
{
	const int DIM = 3;
	vec<DIM>v , w;
	for (int i = 0; i < DIM; i++)
	{
		v[i] = i+1;
		w[i] = (i+1) * (i+1);
	}
		std::cout << "v = " << v << std::endl
			<< "w = " << w << std::endl
			<< "v + w = " << v + w << std::endl
			<< "v - w = " << v - w << std::endl
			<< "v * w = " << v * w << std::endl
			<< "v / w =" << v / w << std::endl
			<< "v . w = " << dot(v, w) << std::endl
			<< "sum of v = " << sum(v) << std::endl
			<< "max of v =" << max(v) << std::endl;
    
	Matrix<DIM, DIM> A, B;

	for (int i = 0; i < DIM; i++)
	{
		A[i] = v + DIM * i;
		B[i] = w + DIM * i;
	}

	std::cout << "A = " << A << std::endl
		<< "B = " << B << std::endl
		<< "A + B = " << A + B << std::endl
		<< "A - B = " << A - B << std::endl
		<< "A * B = " << A * B << std::endl
		<< "A / B =" << A / B << std::endl
		<< "AB = " << dot(A, B) << std::endl
		<< "sum of A = " << sum(A) << std::endl;

	double lambda;
	vec<DIM> temp, x;
	x = v;
	

	for (int i = 0; i < 10; i++)
	{
		temp = dot(A, x);
		lambda = sqrt(dot(temp, temp) / dot(x, x));
		x = temp / max(temp);
	}
	std::cout << "Max Eigenvector of A: " << x << std::endl;
	std::cout << "Max Eigenvalue of A: " << lambda << std::endl;

	return 0;
}
*/
