import weka.core.Instances;
import weka.core.DenseInstance;
import weka.classifiers.functions.LinearRegression;
import weka.core.Attribute;

import java.util.ArrayList;
import java.util.Random;

public class LinearRegressionExample {
    public static void main(String[] args) throws Exception {
        // Create attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("X"));
        attributes.add(new Attribute("Y"));

        // Create dataset
        Instances data = new Instances("LinearRegression", attributes, 100);
        data.setClassIndex(1);

        // Generate sample data
        Random rand = new Random(0);
        for (int i = 0; i < 100; i++) {
            double x = rand.nextDouble();
            double y = 2 + 3 * x + rand.nextGaussian() * 0.1;
            data.add(new DenseInstance(1.0, new double[]{x, y}));
        }

        // Create and train the model
        LinearRegression model = new LinearRegression();
        model.buildClassifier(data);

        // Print results
        System.out.println("Linear Regression Equation:");
        System.out.println(model);

        // Make predictions
        double[] testInstance = new double[]{0.5};
        DenseInstance instance = new DenseInstance(1.0, testInstance);
        instance.setDataset(data);
        double prediction = model.classifyInstance(instance);
        System.out.println("Prediction for X=0.5: " + prediction);
    }
}
