import weka.core.Instances;
import weka.core.DenseInstance;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.classifiers.Evaluation;

import java.util.ArrayList;
import java.util.Random;

public class NeuralNetworkExample {
    public static void main(String[] args) throws Exception {
        // Create attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            attributes.add(new Attribute("X" + i));
        }
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("0");
        classValues.add("1");
        attributes.add(new Attribute("Class", classValues));

        // Create dataset
        Instances data = new Instances("NeuralNetwork", attributes, 100);
        data.setClassIndex(20);

        // Generate sample data
        Random rand = new Random(42);
        for (int i = 0; i < 100; i++) {
            double[] values = new double[21];
            for (int j = 0; j < 20; j++) {
                values[j] = rand.nextDouble();
            }
            values[20] = rand.nextInt(2);
            data.add(new DenseInstance(1.0, values));
        }

        // Create and train the model
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setHiddenLayers("10,5");
        mlp.setTrainingTime(1000);
        mlp.buildClassifier(data);

        // Print the network structure
        System.out.println("Neural Network Structure:");
        System.out.println(mlp);

        // Evaluate the model using cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(mlp, data, 10, new Random(1));

        // Print evaluation results
        System.out.println("Evaluation Results:");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());

        // Make a prediction
        double[] testInstance = new double[20];
        for (int i = 0; i < 20; i++) {
            testInstance[i] = rand.nextDouble();
        }
        DenseInstance instance = new DenseInstance(1.0, testInstance);
        instance.setDataset(data);
        double prediction = mlp.classifyInstance(instance);
        System.out.println("Prediction for test instance: " + data.classAttribute().value((int) prediction));
    }
}
