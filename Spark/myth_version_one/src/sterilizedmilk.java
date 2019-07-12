import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.*;
import org.apache.spark.ml.linalg.Vectors;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.FileSystem;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.io.*;


public class sterilizedmilk {
    //用于存放预警原始数据
    public static List<Row> data_sterilizedmilk = new ArrayList<>();

    //用于存放将要进入算法的训练数据
    public static List<ArrayList<Float>> data_regression = new ArrayList<ArrayList<Float>>();

    //用于存放建模预测之后的测试集数据
    public static List<ArrayList<Float>> data_prediction = new ArrayList<ArrayList<Float>>();

    //存放对应指标的预警权重
    public static List<Float> weight = new ArrayList<Float>();

    public static void main(String[] args) throws Exception{

        //获取各个指标的权重值
        get_weight();

        //获取hive中myth.sterilizedmilk中的数据
        get_hive_data();

        //将数据按照官方文档的libsvm方式写入 0为训练数据 1为测试数据
        //writeFileContext(data_regression, "/home/hadoop/spark-2.4.0/myth/sterilizedmilk_for_regression_train.txt", 0);
        //writeFileContext(data_regression, "/home/hadoop/spark-2.4.0/myth/sterilizedmilk_for_regression_test.txt", 1);

        //获得Dataset类型的训练数据和测试数据
        Dataset<Row> train_data = get_data(0);
        Dataset<Row> test_data = get_data(1);
        //test_data.show();
        List<Row> precition = regression(train_data, test_data);
        write_prediction_hdfs(precition, "/spark/sterilizedmilk/prediction.txt");

    }
    public static void get_weight(){
        weight.add(0.5629f);
        weight.add(0.6077f);
        weight.add(0.1902f);
        weight.add(0.3955f);
        weight.add(0.2871f);
        weight.add(0.0273f);
        weight.add(0.0105f);
        weight.add(0.0146f);
        weight.add(0.0347f);
    }
    public static void get_hive_data() {
        SparkConf conf = new SparkConf().setAppName("sterilizedmilk").setMaster("local");;
        SparkSession spark = SparkSession
                .builder()
                .appName("sterilizedmilk")
                .config(conf)
                .enableHiveSupport()  //支持hive
                .getOrCreate();
        String querySql = "SELECT * FROM myth.sterilizedmilk";
        Dataset<Row> data = spark.sql(querySql);
        data_sterilizedmilk = data.collectAsList();
        for(int i = 0; i < data_sterilizedmilk.size(); i++){
            ArrayList<Float> temp = new ArrayList<Float>();
            float sum = 0;
            for(int j = 2; j < data_sterilizedmilk.get(i).size(); j++){
                temp.add(Float.parseFloat(data_sterilizedmilk.get(i).get(j).toString()));
                sum += Float.parseFloat(data_sterilizedmilk.get(i).get(j).toString()) * weight.get(j-2);
            }
            temp.add(sum);
            data_regression.add(temp);
        }
    }
    public static void writeFileContext(List<ArrayList<Float>> data, String path, int train_or_test) throws Exception {
        File file = new File(path);

        if(!file.exists()) {
            file.createNewFile();
        }
        BufferedWriter writer = new BufferedWriter(new FileWriter(path));

        //存储数据遍历的起始和终止位置
        int start, end;

        if(train_or_test == 0){
            start = 0;
            end = (int)(data_regression.size() * 0.9);
        }
        else{
            start = (int)(data_regression.size() * 0.9);
            end = data_regression.size();
        }
        for(int i = start; i < end; i++){
            String write_string = "";
            write_string += data_regression.get(i).get(9).toString();
            for(int j = 0; j < 9; j++) {
                write_string = write_string + " " + String.valueOf(j + 1) + ":" + data_regression.get(i).get(j).toString();
            }
            writer.write(write_string + "\n");
        }
        writer.close();
    }
    public static Dataset<Row> get_data(int i){
        SparkConf conf = new SparkConf().setAppName("sterilizedmilk").setMaster("local");
        SparkSession spark = SparkSession
                .builder()
                .appName("sterilizedmilk")
                .config(conf)
                .enableHiveSupport()  //支持hive
                .getOrCreate();
        Dataset<Row> data;
        if(i == 0){
            data = spark.read().format("libsvm").load("hdfs://192.168.1.121:9000/spark/sterilizedmilk/sterilizedmilk_for_regression_train.txt");
        }
        else{
            data = spark.read().format("libsvm").load("hdfs://192.168.1.121:9000/spark/sterilizedmilk/sterilizedmilk_for_regression_test.txt");
        }
        return data;
    }
    public static List<Row> regression(Dataset<Row> train_data, Dataset<Row> test_data){
        //实例化LinearRegression对象，并设置参数
        LinearRegression lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8);

        //创建模型对象，并进行训练
        LinearRegressionModel lrModel = lr.fit(train_data);

        //进行预测
        List<Row> prediction = lrModel.transform(test_data).collectAsList();
        //System.out.println(prediction.get(0));

        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());

        return prediction;
    }
    public static void write_prediction_hdfs(List<Row> prediction, String path) throws Exception{

        //获取precition数据
        int interval = (int)(data_regression.size() * 0.9);

        for(int i = interval; i < data_regression.size(); i++){
            ArrayList<Float> temp = data_regression.get(i);
            temp.add(Float.parseFloat(prediction.get(i-interval).get(2).toString()));
            data_prediction.add(temp);
        }
        //将prediction数据写入hdfs中
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS","hdfs://192.168.1.121:9000");
        FileSystem fs = FileSystem.get(conf);
        FSDataOutputStream out = fs.create(new Path(path));
        //写出
        for(int i = 0; i < data_prediction.size(); i++) {
            String re = "";
            for(int j = 0;j<data_prediction.get(i).size();j++) {
                if(j == data_prediction.get(i).size()-1){
                    re = re + data_prediction.get(i).get(j).toString() + "\n";
                }
                else{
                    re = re + data_prediction.get(i).get(j).toString() + " ";
                }
            }
            out.write(re.getBytes("UTF-8"));
        }
        out.close();
    }
}
