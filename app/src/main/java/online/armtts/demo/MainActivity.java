package online.armtts.demo;

import android.os.Bundle;
import android.text.TextUtils;
import android.view.View;
import android.widget.EditText;
import android.widget.RadioGroup;

import androidx.appcompat.app.AppCompatActivity;

import online.armtts.ArmTTS;
import online.armtts.demo.R;

public class MainActivity extends AppCompatActivity {
    private static final String DEFAULT_INPUT_TEXT = "Ողջույն, իմ անունը Գոռ է։";

    private View speakBtn;
    private RadioGroup speedGroup;
    private ArmTTS armTTS;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        armTTS = new ArmTTS(this, "UPDATE_TOKEN_HERE");

        EditText input = findViewById(R.id.input);
        input.setHint(DEFAULT_INPUT_TEXT);

        speedGroup = findViewById(R.id.speed_chooser);
        speedGroup.check(R.id.normal);

        speakBtn = findViewById(R.id.start);
        speakBtn.setOnClickListener(v -> {
                    float speed;
                    switch (speedGroup.getCheckedRadioButtonId()) {
                        case R.id.fast:
                            speed = 1.25F;
                            break;
                        case R.id.slow:
                            speed = 0.75F;
                            break;
                        case R.id.normal:
                        default:
                            speed = 1.0F;
                            break;
                    }

                    String inputText = input.getText().toString();
                    if (TextUtils.isEmpty(inputText)) {
                        inputText = DEFAULT_INPUT_TEXT;
                    }
                    armTTS.play(inputText, speed);
                });
    }
}
