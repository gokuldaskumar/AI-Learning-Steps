import weka.core.Instances;
import weka.core.DenseInstance;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.classifiers.Evaluation;

import java.util.ArrayList;
import java.util.Random;

public class DecisionTreeExample {
    public static void main(String[] args) throws Exception {
        // Create attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("X1"));
        attributes.add(new Attribute("X2"));
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("0");
        classValues.add("1");
        attributes.add(new Attribute("Class", classValues));

        // Create dataset
        Instances data = new Instances("DecisionTree", attributes, 100);
        data.setClassIndex(2);

        // Generate sample data
        Random rand = new Random(42);
        for (int i = 0; i < 100; i++) {
            double x1 = rand.nextDouble();
            double x2 = rand.nextDouble();
            String classValue = (x1 + x2 > 1) ? "1" : "0";
            data.add(new DenseInstance(1.0, new double[]{x1, x2, data.attribute(2).indexOfValue(classValue)}));
        }

        // Create and train the model
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Print the decision tree
        System.out.println("Decision Tree:");
        System.out.println(tree);

        // Evaluate the model using cross-validation
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(tree, data, 10, new Random(1));

        // Print evaluation results
        System.out.println("Evaluation Results:");
        System.out.println(eval.toSummaryString());
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString());
    }
}
