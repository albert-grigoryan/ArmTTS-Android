package online.armtts;

import android.annotation.SuppressLint;
import android.content.Context;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioTrack;
import android.os.AsyncTask;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

class ProcessResult {
    private int rc;
    private int[] sequence;
    private String message;

    public ProcessResult() {
    }

    public int getRc() {
        return rc;
    }

    public void setRc(int rc) {
        this.rc = rc;
    }

    public int[] getSequence() {
        return sequence;
    }

    public void setSequence(int[] sequence) {
        this.sequence = sequence;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}

public class ArmTTS {
    private static final String TAG = "ArmTTS";
    private final static String MODEL1 = "model1.tflite";
    private final static String MODEL2 = "model2.tflite";
    private final static int MAX_LENGTH = 160;

    private final AudioTrack mAudioTrack;
    private final static int FORMAT = AudioFormat.ENCODING_PCM_FLOAT;
    private final static int SAMPLE_RATE = 22000;
    private final static int CHANNEL = AudioFormat.CHANNEL_OUT_MONO;
    private final static int BUFFER_SIZE = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL, FORMAT);

    private String x_RapidAPI_Key;
    private Interpreter module1;
    private Interpreter module2;

    public ArmTTS(Context context, String x_RapidAPI_Key) {
        this.x_RapidAPI_Key = x_RapidAPI_Key;

        // Interpreter options.
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(5);

        String model1 = copyFile(context, MODEL1);
        String model2 = copyFile(context, MODEL2);

        // Load models.
        try {
            module1 = new Interpreter(new File(model1), options);
            module2 = new Interpreter(new File(model2), options);
            Log.i(TAG, "Initialized.");
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Initialization failed.");
        }

        // Initialize player.
        mAudioTrack = new AudioTrack(
                new AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC)
                        .build(),
                new AudioFormat.Builder()
                        .setSampleRate(SAMPLE_RATE)
                        .setEncoding(FORMAT)
                        .setChannelMask(CHANNEL)
                        .build(),
                BUFFER_SIZE,
                AudioTrack.MODE_STREAM, AudioManager.AUDIO_SESSION_ID_GENERATE
        );
        mAudioTrack.play();
    }

    private void playInternal(float[] audio) {
        int index = 0;
        while (index < audio.length) {
            int buffer = Math.min(BUFFER_SIZE, audio.length - index);
            mAudioTrack.write(audio, index, buffer, AudioTrack.WRITE_BLOCKING);
            index += BUFFER_SIZE;
        }
    }

    private ProcessResult process(String text) {
        OkHttpClient client = new OkHttpClient().newBuilder()
                .build();
        MediaType mediaType = MediaType.parse("text/plain");
        RequestBody body = new MultipartBody.Builder().setType(MultipartBody.FORM)
                .addFormDataPart("text", text)
                .build();
        Request request = new Request.Builder()
                .url("https://armtts1.p.rapidapi.com/preprocess")
                .method("POST", body)
                .addHeader("X-RapidAPI-Key", x_RapidAPI_Key)
                .build();
        ProcessResult result = new ProcessResult();
        try {
            Response response = client.newCall(request).execute();
            JSONObject json = new JSONObject(response.body().string());
            if (json.has("ids")) {
                result.setRc(0);
                JSONArray jids = json.getJSONArray("ids");
                int[] ids = new int[jids.length()];
                for (int i = 0; i < ids.length; ++i) {
                    ids[i] = jids.optInt(i);
                }
                result.setSequence(ids);
            } else {
                result.setRc(-1);
                if (json.has("message")) {
                    String message = json.getString("message");
                    result.setMessage(message);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.e(TAG, "Process failed.");
        }
        return result;
    }

    private float[] synthesize(int[] inputIds, float speed) {
        module1.resizeInput(0, new int[]{1, inputIds.length});
        module1.allocateTensors();
        @SuppressLint("UseSparseArrays")
        Map<Integer, Object> outputMap = new HashMap<>();
        FloatBuffer outputBuffer1 = FloatBuffer.allocate(350000);
        outputMap.put(0, outputBuffer1);
        int[][] inputs = new int[1][inputIds.length];
        inputs[0] = inputIds;
        module1.runForMultipleInputsOutputs(
                new Object[]{inputs, new int[1][1], new float[]{speed}, new float[]{1F},
                        new float[]{1F}}, outputMap);
        int size = module1.getOutputTensor(0).shape()[2];
        int[] shape = {1, outputBuffer1.position() / size, size};
        TensorBuffer spectrogram = TensorBuffer.createFixedSize(shape, DataType.FLOAT32);
        float[] outputArray = new float[outputBuffer1.position()];
        outputBuffer1.rewind();
        outputBuffer1.get(outputArray);
        spectrogram.loadArray(outputArray);
        module2.resizeInput(0, spectrogram.getShape());
        module2.allocateTensors();
        FloatBuffer outputBuffer2 = FloatBuffer.allocate(350000);
        module2.run(spectrogram.getBuffer(), outputBuffer2);
        float[] audioArray = new float[outputBuffer2.position()];
        outputBuffer2.rewind();
        outputBuffer2.get(audioArray);
        return audioArray;
    }

    private Boolean isInitialized() {
        return module1 != null && module2 != null;
    }

    private List<String> tokenize(String text) {
        String[] tmp_tokens = text.split("[:։]");
        List<String> tokens = new ArrayList<>();
        for (String token : tmp_tokens) {
            if (token.length() > MAX_LENGTH) {
                String tmp_string = "";
                String[] words = text.split(" ");
                for (String word : words) {
                    if (tmp_string.length() + word.length() < MAX_LENGTH/2) {
                        tmp_string += " " + word;
                    } else {
                        if (!tmp_string.isEmpty()) {
                            tokens.add(tmp_string.trim());
                        }
                        tmp_string = word;
                    }
                }
                if (!tmp_string.isEmpty()) {
                    tokens.add(tmp_string.trim());
                }
            } else {
                tokens.add(token.trim());
            }
        }
        return tokens;
    }

    public void play(String text, float speed) {
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                if (!isInitialized()) {
                    Log.e(TAG, "The initialization has been failed. Please check the logs for more details.");
                    return;
                }
                List<String> sentences = tokenize(text);
                ArrayList<float[]> audios = new ArrayList<>();
                for (String sentence : sentences) {
                    ProcessResult processResult = process(sentence);
                    if (processResult.getRc() == 0) {
                        float reversedSpeed = 2 - speed;
                        int[] sequence = processResult.getSequence();
                        if (sequence != null && sequence.length != 0) {
                            float[] audio = synthesize(sequence, reversedSpeed);
                            audios.add(audio);
                        }
                    } else {
                        Log.e(TAG, processResult.getMessage());
                    }
                }
                for (float[] audio : audios) {
                    playInternal(audio);
                }
            }
        });
    }

    private String copyFile(Context context, String strOutFileName) {
        File file = context.getFilesDir();
        String tmpFile = file.getAbsolutePath() + "/" + strOutFileName;
        File f = new File(tmpFile);
        if (f.exists()) { return f.getAbsolutePath(); }
        try (OutputStream myOutput = new FileOutputStream(f);
             InputStream myInput = context.getAssets().open(strOutFileName)) {
            byte[] buffer = new byte[1024];
            int length = myInput.read(buffer);
            while (length > 0) {
                myOutput.write(buffer, 0, length);
                length = myInput.read(buffer);
            }
            myOutput.flush();
        } catch (Exception e) {
            Log.e(TAG, "Failed to copy", e);
        }
        return f.getAbsolutePath();
    }
}
