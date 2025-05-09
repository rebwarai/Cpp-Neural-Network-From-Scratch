#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <stdexcept>
#include <chrono>

using namespace std;

namespace Activation{
    inline double relu(double x) { return max(0.0, x); }
    inline double reluDerivative(double x) { return (x > 0) ? 1.0 : 0.0; }
    inline double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
    inline double sigmoidDerivative(double x){
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
}

class Matrix{
private:
    vector<vector<double>> data;
    size_t rows, cols;

public: 
    Matrix(size_t r,size_t c):rows(r),cols(c){
        data.resize(r, vector<double>(c, 0.0));
    }

    double &operator()(size_t i, size_t j) { return data[i][j]; }
    const double &operator()(size_t i, size_t j) const { return data[i][j]; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
};

class NeuralNetwork
{
private:
    vector<int> layerSizes;
    Matrix weights1, weights2, weights3;
    vector<double> bias1, bias2, bias3;
    mt19937 gen;

    void initializeWeights(){
        normal_distribution<> dist(0.0, 1.0);
        double scale1 = sqrt(2.0 / layerSizes[0]);
        double sclae2 = sqrt(2.0 / layerSizes[1]);
        double sclae3 = sqrt(2.0 / layerSizes[2]);

        for (int i = 0; i < weights1.getRows();i++)
            for (int j = 0; j < weights1.getCols(); j++)
                weights1(i, j) = dist(gen) * scale1;

        for (int i = 0; i < weights2.getRows();i++)
            for (int j = 0; j < weights2.getCols();j++)
                weights2(i, j) = dist(gen) * sclae2;

        for (int i = 0; i < weights3.getRows();i++)
            for (int j = 0; j < weights3.getCols();j++)
                weights3(i, j) = dist(gen) * sclae3;

        fill(bias1.begin(), bias1.end(), 0.0);
        fill(bias2.begin(), bias2.end(), 0.0);
        fill(bias3.begin(), bias3.end(), 0.0);
    }

public:
    NeuralNetwork(int inputSize,int hidden1Size,int hidden2Szie,int outputSize)
    : layerSizes{inputSize,hidden1Size,hidden2Szie,outputSize},
        weights1(inputSize,hidden1Size),
        weights2(hidden1Size,hidden2Szie),
        weights3(hidden2Szie,outputSize),
        bias1(hidden1Size),bias2(hidden2Szie),bias3(outputSize),
        gen(random_device{}())
    {
        if(inputSize <= 0 || hidden1Size <= 0 || hidden2Szie <= 0 || outputSize <= 0)
        {
            throw invalid_argument("Layer sizes must be positive !");
        }

        initializeWeights();
    }

    vector<double> forward( const vector<double> &input){
        if(input.size() != layerSizes[0])
        {
            throw runtime_error("Input size mismatch !");
        }

        vector<double> hidden1(layerSizes[1]);
        for (int j = 0; j < layerSizes[1];j++)
        {
            double sum = bias1[j];
            for (int i = 0; i < layerSizes[0];i++)
            {
                sum += input[i] * weights1(i, j);
            }

            hidden1[j] = Activation::relu(sum);
        }

        vector<double> hidden2(layerSizes[2]);
        for (int j = 0; j < layerSizes[2]; j++)
        {
            double sum = bias2[j];
            for (int i = 0; i < layerSizes[1]; i++)
            {
                sum += hidden1[i] * weights2(i, j);
            }

            hidden2[j] = Activation::relu(sum);
        }

        vector<double> output(layerSizes[3]);
        for (int j = 0; j < layerSizes[3]; j++)
        {
            double sum = bias3[j];
            for (int i = 0; i < layerSizes[2]; i++)
            {
                sum += hidden2[i] * weights3(i, j);
            }

            output[j] = Activation::sigmoid(sum);
        }

        return output;
    }

    void train(const vector<vector<double>> &inputs,
                const vector<vector<double>> &targets,
                double learningRate , int epochs)
    {
        if(inputs.size() != targets.size())
        {
            throw runtime_error("Input and Target sizes don't match !");
        }

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0.0;

            for (size_t k = 0; k < inputs.size();k++)
            {
                vector<double> hidden1(layerSizes[1]);
                vector<double> hidden1Pre(layerSizes[1]);
                for (int j = 0; j < layerSizes[1]; j++)
                {
                    double sum = bias1[j];
                    for (int i = 0; i < layerSizes[0];i++)
                    {
                        sum += inputs[k][i] * weights1(i, j);
                    }
                    hidden1Pre[j] = sum;
                    hidden1[j] = Activation::relu(sum);
                }

                vector<double> hidden2(layerSizes[2]);
                vector<double> hidden2Pre(layerSizes[2]);
                for (int j = 0; j < layerSizes[2];j++)
                {
                    double sum = bias2[j];
                    for (int i = 0; i < layerSizes[1];i++)
                    {
                        sum += hidden1[i] * weights2(i, j);
                    }
                    hidden2Pre[j] = sum;
                    hidden2[j] = Activation::relu(sum);
                }

                vector<double> output(layerSizes[3]);
                vector<double> outputPre(layerSizes[3]);
                for (int j = 0; j < layerSizes[3];j++)
                {
                    double sum = bias3[j];
                    for (int i = 0; i < layerSizes[2];i++)
                    {
                        sum += hidden2[i] * weights3(i, j);
                    }
                    outputPre[j] = sum;
                    output[j] = Activation::sigmoid(sum);
                }

                for (int j = 0; j < layerSizes[3]; j++)
                {
                    double error = targets[k][j] - output[j];
                    totalError += error * error;
                }

                vector<double> outputGradients(layerSizes[3]);
                for (int j = 0; j < layerSizes[3];j++)
                {
                    outputGradients[j] = (output[j] - targets[k][j]) *
                                         Activation::sigmoidDerivative(outputPre[j]);
                }

                vector<double> hidden2Gradients(layerSizes[2]);
                for (int i = 0; i < layerSizes[2]; i++)
                {
                    double error = 0;
                    for (int j = 0; j < layerSizes[3];j++)
                    {
                        error += outputGradients[j] * weights3(i, j);
                    }

                    hidden2Gradients[i] = error * Activation::reluDerivative(hidden2Pre[i]);
                }

                vector<double> hidden1Gradients(layerSizes[1]);
                for (int i = 0; i < layerSizes[1];i++)
                {
                    double error = 0;
                    for (int j = 0; j < layerSizes[2];j++)
                    {
                        error += hidden2Gradients[j] * weights2(i, j);
                    }

                    hidden1Gradients[i] = error * Activation::reluDerivative(hidden1Pre[i]);
                }

                for (int i = 0; i < layerSizes[2];i++)
                {
                    for (int j = 0; j < layerSizes[3];j++)
                    {
                        weights3(i, j) -= learningRate * outputGradients[j] * hidden2[i];
                    }
                }

                for (int j = 0; j < layerSizes[3];j++)
                {
                    bias3[j] -= learningRate * outputGradients[j];
                }

                for (int i = 0; i < layerSizes[1];i++)
                {
                    for (int j = 0; j < layerSizes[2];j++)
                    {
                        weights2(i, j) -= learningRate * hidden2Gradients[j] * hidden1[i];
                    }
                }

                for (int j = 0; j < layerSizes[2];j++)
                {
                    bias2[j] -= learningRate * hidden2Gradients[j];
                }

                for (int i = 0; i < layerSizes[0];i++)
                {
                    for (int j = 0; j < layerSizes[1];j++)
                    {
                        weights1(i, j) -= learningRate * hidden1Gradients[j] * inputs[k][i];
                    }
                }

                for (int j = 0; j < layerSizes[1];j++)
                {
                    bias1[j] -= learningRate * hidden1Gradients[j];
                }
            }

            if(epoch % 100 == 0)
            {
                cout << "EPOCH : " << epoch << "  MSE : "
                     << totalError / inputs.size() << "\n";
            }
        }
    }
};

int main(){

    try{

        NeuralNetwork nn(2, 8, 4, 1);
        mt19937 gen(random_device{}());
        uniform_real_distribution<> dist(-2.0, 2.0);
        const int numSamples = 1000;
        vector<vector<double>> inputs(numSamples);
        vector<vector<double>> targets(numSamples);

        for (int i = 0; i < numSamples;i++)
        {
            double x = dist(gen);
            double y = dist(gen);
            inputs[i] = {x, y};
            double distance = sqrt(x * x + y * y);
            targets[i] = {distance < 1.0 ? 1.0 : 0.0};
        }

        auto start = chrono::high_resolution_clock::now();
        nn.train(inputs, targets, 0.01, 1000);
        auto end = chrono::high_resolution_clock::now();
        cout << "Training Time : "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";
        vector<vector<double>> testPoints = {
            {0.0, 0.0},
            {1.0, 1.0},
            {0.5, 0.5},
            {2.0, 0.0}};

        cout << "\n Test Results (1 = inside , 0 = outside) : \n";
        for(const auto &point : testPoints)
        {
            auto output = nn.forward(point);
            double actual = sqrt(point[0] * point[0] + point[1] * point[1]) < 1.0 ? 1.0 : 0.0;
            cout << " point ( " << point[0] << "," << point[1] << " ) :"
                 << output[0] << " ( actual : " << actual
                 << " , error : " << abs(output[0] - actual) << " ) \n";
        }
    }catch(const exception &e){
        cerr << " ERROR : " << e.what() << endl;
        return 1;
    }

    return 0;
}