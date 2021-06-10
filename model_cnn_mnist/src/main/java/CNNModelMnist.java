import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import play.libs.F;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class CNNModelMnist {
    public static void main(String[] args) throws IOException, InterruptedException {
        long seed=1234;
        double learningRate=0.001;
        long height=28;
        long width=28;
        long depth=1;
        int outPutSize=10;
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(learningRate))
                .list()
                .setInputType(InputType.convolutionalFlat(height,width,depth))
                .layer(0,new ConvolutionLayer.Builder()
                        .nIn(depth)
                        .nOut(20)
                        .activation(Activation.RELU)
                        .kernelSize(5,5)
                        .stride(1,1)
                        .build())
                .layer(1,new SubsamplingLayer.Builder()
                        .kernelSize(2,2)
                        .stride(2,2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(2,new ConvolutionLayer.Builder()
                        .nOut(50)
                        .activation(Activation.RELU)
                        .stride(1,1)
                        .build())
                .layer(3,new SubsamplingLayer.Builder()
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4,new DenseLayer.Builder()
                        .nOut(500)
                        .activation(Activation.RELU)
                        .build())
                .layer(5,new OutputLayer.Builder()
                        .nOut(outPutSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .build())
                .build();
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        model.init();


        String path=System.getProperty("user.home")+"/mnist";
        File fileTrain=new File(path+"/training");
        FileSplit fileSplitTrain=new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS,new Random(seed));
        RecordReader recordReaderTrain=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
        recordReaderTrain.initialize(fileSplitTrain);
        int batchSize=54;
        DataSetIterator dataSetIteratorTrain=new RecordReaderDataSetIterator(recordReaderTrain,batchSize,1,outPutSize);
        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        dataSetIteratorTrain.setPreProcessor(scaler);

        System.out.println("Entrainement du model");
        UIServer uiServer=UIServer.getInstance();
        StatsStorage statsStorage=new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));

        int numEpoch=1;
        for (int i = 0; i <numEpoch ; i++) {
            model.fit(dataSetIteratorTrain);
        }
        System.out.println("Evaluation du model");
        File fileTest=new File(path+"/testing");
        FileSplit fileSplitTest=new FileSplit(fileTrain, NativeImageLoader.ALLOWED_FORMATS,new Random(seed));
        RecordReader recordReaderTest=new ImageRecordReader(height,width,depth,new ParentPathLabelGenerator());
        recordReaderTest.initialize(fileSplitTest);
        DataSetIterator dataSetIteratorTest=new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outPutSize);
        DataNormalization scalerTest=new ImagePreProcessingScaler(0,1);
        dataSetIteratorTrain.setPreProcessor(scalerTest);

        Evaluation evaluation=new Evaluation();
        while (dataSetIteratorTest.hasNext()){
            DataSet dataSet=dataSetIteratorTest.next();
            INDArray features=dataSet.getFeatures();
            INDArray targetLabels=dataSet.getLabels();

            INDArray predicted=model.output(features);
            evaluation.eval(predicted,targetLabels);
        }
        System.out.println(evaluation.stats());



    }
}
