package com.example.imageclassificationlivefeed

//import android.annotation.SuppressLint
//import android.content.res.AssetManager
//import android.graphics.Bitmap
//import android.util.Log
//import org.tensorflow.lite.Interpreter
//import java.io.BufferedReader
//import java.io.FileInputStream
//import java.io.IOException
//import java.io.InputStreamReader
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.nio.MappedByteBuffer
//import java.nio.channels.FileChannel
//import java.util.*
//
//class Classifier(assetManager: AssetManager, modelPath: String, labelPath: String, inputSize: Int) {
//    private var interpreter: Interpreter
//    private var lableList: List<String>
//    private val INPUT_SIZE: Int = inputSize
//    private val PIXEL_SIZE: Int = 3
//    private val IMAGE_MEAN = 0
//    private val IMAGE_STD = 255.0f
//    private val MAX_RESULTS = 3
//    private val THRESHOLD = 0.1f
//
//    data class Recognition(
//            var id: String = "",
//            var title: String = "",
//            var confidence: Float = 0F
//    )  {
//        override fun toString(): String {
//            return "Title = $title, Confidence = $confidence)"
//        }
//    }
//
//    init {
//        val options = Interpreter.Options()
//        options.setNumThreads(5)
//        options.setUseNNAPI(true)
//        interpreter = Interpreter(loadModelFile(assetManager, modelPath), options)
//        lableList = loadLabelList(assetManager, labelPath)
//    }
//
//    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
//        val fileDescriptor = assetManager.openFd(modelPath)
//        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
//        val fileChannel = inputStream.channel
//        val startOffset = fileDescriptor.startOffset
//        val declaredLength = fileDescriptor.declaredLength
//        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
//    }
//
//    private fun loadLabelList(assetManager: AssetManager, labelPath: String): List<String> {
//        return assetManager.open(labelPath).bufferedReader().useLines { it.toList() }
//
//    }
//
//    /**
//     * Returns the result after running the recognition with the help of interpreter
//     * on the passed bitmap
//     */
//    fun recognizeImage(bitmap: Bitmap): List<Recognition> {
//        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
//        val byteBuffer = convertBitmapToByteBuffer(scaledBitmap)
//        val result = Array(1) { FloatArray(lableList.size) }
//        interpreter.run(byteBuffer, result)
//        return getSortedResult(result)
//    }
//
//
//    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
//        val byteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE)
//        byteBuffer.order(ByteOrder.nativeOrder())
//        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
//
//        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
//        var pixel = 0
//        for (i in 0 until INPUT_SIZE) {
//            for (j in 0 until INPUT_SIZE) {
//                val input = intValues[pixel++]
//
//                byteBuffer.putFloat((((input.shr(16)  and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
//                byteBuffer.putFloat((((input.shr(8) and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
//                byteBuffer.putFloat((((input and 0xFF) - IMAGE_MEAN) / IMAGE_STD))
//            }
//        }
//        return byteBuffer
//    }
//
//    private fun getSortedResult(labelProbArray: Array<FloatArray>): List<Classifier.Recognition> {
//        Log.d("Classifier", "List Size:(%d, %d, %d)".format(labelProbArray.size,labelProbArray[0].size,lableList.size))
//
//        val pq = PriorityQueue(
//                MAX_RESULTS,
//                Comparator<Classifier.Recognition> {
//                    (_, _, confidence1), (_, _, confidence2)
//                    -> java.lang.Float.compare(confidence1, confidence2) * -1
//                })
//
//        for (i in lableList.indices) {
//            val confidence = labelProbArray[0][i]
//            if (confidence >= THRESHOLD) {
//                pq.add(Classifier.Recognition("" + i,
//                        if (lableList.size > i) lableList[i] else "Unknown", confidence)
//                )
//            }
//        }
//        Log.d("Classifier", "pqsize:(%d)".format(pq.size))
//
//        val recognitions = ArrayList<Classifier.Recognition>()
//        val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
//        for (i in 0 until recognitionsSize) {
//            recognitions.add(pq.poll())
//        }
//        return recognitions
//    }
//
//}


import android.app.Activity
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.TensorProcessor
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.label.TensorLabel
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

import java.nio.MappedByteBuffer
import java.util.*

import java.lang.Math.min;
import java.util.ArrayList
import java.util.Comparator
import java.util.PriorityQueue
/** A classifier specialized to label images using TensorFlow Lite.  */
class Classifier(activity: Activity?, device: Device?, numThreads: Int, modelPath: String?, labelPath: String?) {
    private var gpuDelegate: GpuDelegate? = null
    /** The model type used for classification.  */
    /** The runtime device type used for executing classification.  */
    enum class Device {
        CPU, NNAPI, GPU
    }
    /** The loaded TensorFlow Lite model.  */
    /** Get the image size along the x axis.  */
    /** Image size along the x axis.  */
    val imageSizeX: Int

    /** Get the image size along the y axis.  */
    /** Image size along the y axis.  */
    val imageSizeY: Int

    /** Optional NNAPI delegate for accleration.  */
    private var nnApiDelegate: NnApiDelegate? = null

    /** An instance of the driver class to run model inference with Tensorflow Lite.  */
    protected var tflite: Interpreter?

    /** Options for configuring the Interpreter.  */
    private val tfliteOptions = Interpreter.Options()

    /** Labels corresponding to the output of the vision model.  */
    private val labels: List<String>

    /** Input image TensorBuffer.  */
    private var inputImageBuffer: TensorImage

    /** Output probability TensorBuffer.  */
    private val outputProbabilityBuffer: TensorBuffer

    /** Processer to apply post processing of the output probability.  */
    private val probabilityProcessor: TensorProcessor
    /**
     * Creates a classifier with the provided configuration.
     *
     * @param activity The current Activity.
     *
     * @param device The device to use for classification.
     * @param numThreads The number of threads to use for classification.
     * @return A classifier with the desired configuration.
     */
    /** An immutable result returned by a Classifier describing what was recognized.  */
    class Recognition(
            /**
             * A unique identifier for what has been recognized. Specific to the class, not the instance of
             * the object.
             */
            val id: String?,
            /** Display name for the recognition.  */
            val title: String?,
            /**
             * A sortable score for how good the recognition is relative to others. Higher should be better.
             */
            val confidence: Float?,
            /** Optional location within the source image for the location of the recognized object.  */
            private var location: RectF?) {

        fun getLocation(): RectF {
            return RectF(location)
        }

        fun setLocation(location: RectF?) {
            this.location = location
        }

        override fun toString(): String {
            var resultString = ""
            if (id != null) {
                resultString += "[$id] "
            }
            if (title != null) {
                resultString += "$title "
            }
            if (confidence != null) {
                resultString += String.format("(%.1f%%) ", confidence * 100.0f)
            }
            if (location != null) {
                resultString += location.toString() + " "
            }
            return resultString.trim { it <= ' ' }
        }

    }

    /** Runs inference and returns the classification results.  */
    fun recognizeImage(bitmap: Bitmap, sensorOrientation: Int): List<Recognition> {
        // Logs this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage")
        Trace.beginSection("loadImage")
        val startTimeForLoadImage = SystemClock.uptimeMillis()
        inputImageBuffer = loadImage(bitmap, sensorOrientation)
        val endTimeForLoadImage = SystemClock.uptimeMillis()
        Trace.endSection()
        Log.v(TAG, "Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage))

        // Runs the inference call.
        Trace.beginSection("runInference")
        val startTimeForReference = SystemClock.uptimeMillis()
        tflite!!.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind())
        val endTimeForReference = SystemClock.uptimeMillis()
        Trace.endSection()
        Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference))

        // Gets the map of label and probability.
        val labeledProbability: Map<String, Float> = TensorLabel(labels, probabilityProcessor.process(outputProbabilityBuffer))
                .getMapWithFloatValue()
        Trace.endSection()

        // Gets top-k results.
        return getTopKProbability(labeledProbability)
    }

    /** Closes the interpreter and model to release resources.  */
    fun close() {
        if (tflite != null) {
            tflite!!.close()
            tflite = null
        }
        if (gpuDelegate != null) {
            gpuDelegate!!.close()
            gpuDelegate = null
        }
        if (nnApiDelegate != null) {
            nnApiDelegate!!.close()
            nnApiDelegate = null
        }
    }

    /** Loads input image, and applies preprocessing.  */
    private fun loadImage(bitmap: Bitmap, sensorOrientation: Int): TensorImage {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap)

        // Creates processor for the TensorImage.
        val cropSize = Math.min(bitmap.width, bitmap.height)
        val numRotation = sensorOrientation / 90
        // TODO(b/143564309): Fuse ops inside ImageProcessor.
        val imageProcessor: ImageProcessor = ImageProcessor.Builder()
                .add(ResizeWithCropOrPadOp(cropSize, cropSize)) // TODO(b/169379396): investigate the impact of the resize algorithm on accuracy.
                // To get the same inference results as lib_task_api, which is built on top of the Task
                // Library, use ResizeMethod.BILINEAR.
                .add(ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(Rot90Op(numRotation))
                .add(preprocessNormalizeOp)
                .build()
        return imageProcessor.process(inputImageBuffer)
    }

    protected val preprocessNormalizeOp: TensorOperator
        protected get() = NormalizeOp(IMAGE_MEAN, IMAGE_STD)

    protected val postprocessNormalizeOp: TensorOperator
        protected get() = NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD)

    companion object {
        const val TAG = "ClassifierWithSupport"
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        private const val PROBABILITY_MEAN = 0.0f
        private const val PROBABILITY_STD = 1.0f

        /** Number of results to show in the UI.  */
        private const val MAX_RESULTS = 3

        /** Gets the top-k results.  */
        private fun getTopKProbability(labelProb: Map<String, Float>): List<Recognition> {
            // Find the best classifications.
            val pq = PriorityQueue<Recognition>(
                    MAX_RESULTS,
                    object : Comparator<Recognition?> {
                        override fun compare(o1: Recognition?, o2: Recognition?): Int {
                            return java.lang.Float.compare(o2?.confidence!!, o1?.confidence!!)
                        }
                    })
            for ((key, value) in labelProb) {
                pq.add(Recognition("" + key, key, value, null))
                Log.d("tryResult",key+"     "+value);
            }
            val recognitions = ArrayList<Recognition>()
            val recognitionsSize = Math.min(pq.size, MAX_RESULTS)
            for (i in 0 until recognitionsSize) {
                recognitions.add(pq.poll())

            }
            return recognitions
        }
    }

    /** Initializes a `Classifier`.  */
    init {
        val tfliteModel: MappedByteBuffer = FileUtil.loadMappedFile(activity!!, modelPath!!)
        when (device) {
            Device.NNAPI -> {
                nnApiDelegate = NnApiDelegate()
                tfliteOptions.addDelegate(nnApiDelegate)
            }
            Device.GPU -> {
                gpuDelegate = GpuDelegate()
                tfliteOptions.addDelegate(gpuDelegate)
            }
            Device.CPU -> tfliteOptions.setUseXNNPACK(true)
        }
        tfliteOptions.setNumThreads(numThreads)
        tflite = Interpreter(tfliteModel, tfliteOptions)

        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, labelPath!!)

        // Reads type and shape of input and output tensors, respectively.
        val imageTensorIndex = 0
        val imageShape = tflite!!.getInputTensor(imageTensorIndex).shape() // {1, height, width, 3}
        imageSizeY = imageShape[1]
        imageSizeX = imageShape[2]
        val imageDataType = tflite!!.getInputTensor(imageTensorIndex).dataType()
        val probabilityTensorIndex = 0
        val probabilityShape = tflite!!.getOutputTensor(probabilityTensorIndex).shape() // {1, NUM_CLASSES}
        val probabilityDataType = tflite!!.getOutputTensor(probabilityTensorIndex).dataType()

        // Creates the input tensor.
        inputImageBuffer = TensorImage(imageDataType)

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType)

        // Creates the post processor for the output probability.
        probabilityProcessor = TensorProcessor.Builder().add(postprocessNormalizeOp).build()
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.")
    }
}
