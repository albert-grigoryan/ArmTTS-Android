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

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

class ProcessResult {
    private int rc;
    private long[] sequence;
    private String message;

    public ProcessResult() {
    }

    public int getRc() {
        return rc;
    }

    public void setRc(int rc) {
        this.rc = rc;
    }

    public long[] getSequence() {
        return sequence;
    }

    public void setSequence(long[] sequence) {
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
    private final static String MODEL = "arm-gor.onnx";
    private final static int MAX_LENGTH = 140;

    private final AudioTrack mAudioTrack;
    private final static int FORMAT = AudioFormat.ENCODING_PCM_FLOAT;
    private final static int SAMPLE_RATE = 44100;
    private final static int CHANNEL = AudioFormat.CHANNEL_OUT_MONO;
    private final static int BUFFER_SIZE = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL, FORMAT);

    private String x_RapidAPI_Key;
    private OrtEnvironment ortEnv = null;
    private OrtSession ortSession = null;

    private static byte[] readFileToBytes(String filePath) throws IOException {
        File file = new File(filePath);
        byte[] bytes = new byte[(int) file.length()];
        try(FileInputStream fis = new FileInputStream(file)){
            fis.read(bytes);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return bytes;
    }

    public ArmTTS(Context context, String x_RapidAPI_Key) {
        this.x_RapidAPI_Key = x_RapidAPI_Key;
        String model_file = copyFile(context, MODEL);
        try {
            ortEnv = OrtEnvironment.getEnvironment();
            byte[] model = readFileToBytes(model_file);
            ortSession = ortEnv.createSession(model);
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
                .url("https://armtts1.p.rapidapi.com/v2/preprocess")
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
                long[] ids = new long[jids.length()];
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

    public float[] synthesize(long[][] inputIds, float speed) {
        HashMap<String, OnnxTensor> inputs = new HashMap<>();
        OrtSession.Result result = null;
        try {
            long[] inputLengthsArray = new long[]{inputIds.length};
            for (int i = 0; i < inputIds.length; ++i) {
                inputLengthsArray[i] = inputIds[i].length;
            }
            float[] scalesArray = new float[]{0.333f, speed, 0.0f};
            OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputIds);
            OnnxTensor inputLengthsTensor = OnnxTensor.createTensor(ortEnv, inputLengthsArray);
            OnnxTensor scalesArrayTensor = OnnxTensor.createTensor(ortEnv, scalesArray);
            inputs.put("input", inputTensor);
            inputs.put("input_lengths", inputLengthsTensor);
            inputs.put("scales", scalesArrayTensor);
            result = ortSession.run(inputs);
            System.out.print(result);
        } catch (OrtException e) {
            e.printStackTrace();
        }

        if (result != null) {
            Optional<OnnxValue> resultValue = result.get("output");
            try {
                @SuppressLint({"NewApi", "LocalSuppress"}) float[][][][] resultArray = (float[][][][])resultValue.get().getValue();
                List<Float> resultList = new ArrayList(resultArray[0][0].length);
                for (int i = 0; i < resultArray.length; i++) {
                    for (int j = 0; j < resultArray[i].length; j++) {
                        for (int k = 0; k < resultArray[i][j].length; k++) {
                            for (int q = 0; q < resultArray[i][j][k].length; q++) {
                                resultList.add(resultArray[i][j][k][q]);
                            }
                        }
                    }
                }
                float[] mels = new float[resultList.size()];
                for (int i = 0; i < resultList.size(); ++i) {
                    mels[i] = resultList.get(i);
                }
                return mels;
            } catch (OrtException e) {
                e.printStackTrace();
            }
        }

        return null;
    }

    private Boolean isInitialized() {
        return ortSession != null;
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
                        long[] sequence = processResult.getSequence();
                        if (sequence != null && sequence.length != 0) {
                            float[] audio = synthesize(new long[][]{sequence}, reversedSpeed);
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
