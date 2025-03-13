package org.example;

import ai.djl.Application;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.opencsv.CSVReader;
import com.opencsv.CSVWriter;
import com.opencsv.exceptions.CsvValidationException;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

class SentimentAnalysisCSV {
    public static void main(String[] args) {
        String inputFilePath = "src/main/resources/input_reviews.csv";  // Input CSV file
        String outputFilePath = "src/main/resources/output_sentiment_results.csv";  // Output CSV file

        try {
            // Load the pre-trained sentiment analysis model
            String modelName = "distilbert-base-uncased-finetuned-sst-2-english";
            Criteria<String, Classifications> criteria = Criteria.builder()
                    .optApplication(Application.NLP.SENTIMENT_ANALYSIS)  // Define the task
                    .setTypes(String.class, Classifications.class)  // Input and Output types
                    .optEngine("PyTorch")  // Use PyTorch backend
                    .build();

            ZooModel<String, Classifications> model = ModelZoo.loadModel(criteria);

            // Read CSV and analyze sentiment
            processCSV(model, inputFilePath, outputFilePath);

            System.out.println("✅ Sentiment analysis completed! Check the output file: " + outputFilePath);

        } catch (IOException | ModelException e) {
            e.printStackTrace();
        }
    }

    private static void processCSV(ZooModel<String, Classifications> model, String inputFile, String outputFile) {
        try (
                FileReader fileReader = new FileReader(inputFile);
                CSVReader csvReader = new CSVReader(fileReader);
                FileWriter fileWriter = new FileWriter(outputFile);
                CSVWriter csvWriter = new CSVWriter(fileWriter);
                Predictor<String, Classifications> predictor = model.newPredictor()
        ) {
            // Read CSV header
            String[] header = csvReader.readNext();
            if (header != null) {
                String[] newHeader = Arrays.copyOf(header, header.length + 2);
                newHeader[newHeader.length - 2] = "Sentiment";
                newHeader[newHeader.length - 1] = "Confidence";
                csvWriter.writeNext(newHeader);
            }

            // Process each row
            String[] row;
            while ((row = csvReader.readNext()) != null) {
                String reviewText = row[0]; // Assuming first column is text

                // Perform sentiment analysis
                Classifications result = predictor.predict(reviewText);
                Classifications.Classification bestResult = result.best();

                // Debugging: Print the analysis results
                System.out.println("Review: " + reviewText);
                System.out.println("Predicted Sentiment: " + bestResult.getClassName() + " (" +
                        String.format("%.2f", bestResult.getProbability() * 100) + "%)");

                // Write to CSV file
                String[] newRow = Arrays.copyOf(row, row.length + 2);
                newRow[newRow.length - 2] = bestResult.getClassName();
                newRow[newRow.length - 1] = String.format("%.2f", bestResult.getProbability() * 100) + "%";
                csvWriter.writeNext(newRow);
            }

            // ✅ Flush and Close to Ensure Data is Saved
            csvWriter.flush();
            csvWriter.close();
            System.out.println("✅ Sentiment analysis results successfully written to " + outputFile);

        } catch (IOException | TranslateException | CsvValidationException e) {
            e.printStackTrace();
        }
    }
}
