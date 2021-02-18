package com.example.myapplication52;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;


public class MainActivity extends AppCompatActivity {
    private final int CAMERA_REQUEST_CODE = 100;
    ImageView imgView;
    Button btnPhoto;
    Button btn_repeat;
    protected static final int CameraRequest = 1;
    public File finalFile;
    private TextToSpeech mTTS;
    TextView txtview;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imgView = findViewById(R.id.imageView);
        btnPhoto = findViewById(R.id.btn_photo);
        btn_repeat=findViewById(R.id.repeat);
        txtview=(findViewById(R.id.textView));
        btn_repeat.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                speak();
            }
        });
        mTTS = new TextToSpeech(this, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = mTTS.setLanguage(Locale.US);
                    if (result == TextToSpeech.LANG_MISSING_DATA
                            || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Toast.makeText(MainActivity.this, "Language not supported", Toast.LENGTH_SHORT).show();
                        Log.e("TTS", "Language not supported");
                    } else {
                       // mButtonSpeak.setEnabled(true);
                    }
                } else {
                    Toast.makeText(MainActivity.this, "Initialization failed", Toast.LENGTH_SHORT).show();
                    Log.e("TTS", "Initialization failed");
                }
            }
        });


        requeststorage();

     //   t1 = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {

//            @Override
//            public void onInit(int status) {
//                if (status != TextToSpeech.ERROR) {
//                    t1.setLanguage(Locale.UK);
//                }
//            }
//        });
        btnPhoto.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                Intent camIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                try {

                    startActivityForResult(camIntent, CameraRequest);
                    //startActivity(new Intent(activity_main.this, second.class));

                } catch (ActivityNotFoundException e) {

                    Toast.makeText(MainActivity.this, "برنامه دوربین پیدا نشد", Toast.LENGTH_SHORT).show();

                }

            }
        });
    }


        // uploadFoto();


    private void speak() {
        String text = txtview.getText().toString();
//        float pitch = (float) mSeekBarPitch.getProgress() / 50;
//        if (pitch < 0.1) pitch = 0.1f;
//        float speed = (float) mSeekBarSpeed.getProgress() / 50;
//        if (speed < 0.1) speed = 0.1f;
//        mTTS.setPitch(pitch);
//        mTTS.setSpeechRate(speed);
        mTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }
    private void speak1() {
        String text = "Please wait a moment";
//        float pitch = (float) mSeekBarPitch.getProgress() / 50;
//        if (pitch < 0.1) pitch = 0.1f;
//        float speed = (float) mSeekBarSpeed.getProgress() / 50;
//        if (speed < 0.1) speed = 0.1f;
//        mTTS.setPitch(pitch);
//        mTTS.setSpeechRate(speed);
        mTTS.speak(text, TextToSpeech.QUEUE_FLUSH, null);
    }
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == CameraRequest) {

            Bitmap resultPhoto = (Bitmap) data.getExtras().get("data");
            imgView.setImageBitmap(resultPhoto);
            Uri tempUri = getImageUri(getApplicationContext(), resultPhoto);
            finalFile = new File(getRealPathFromURI(tempUri));
           // Toast.makeText(this, finalFile.toString(), Toast.LENGTH_LONG).show();
            requeststorage();
            uploadFoto();
            speak1();
        }

    }

    public Uri getImageUri(Context inContext, Bitmap inImage) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        inImage.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
        String path = MediaStore.Images.Media.insertImage(inContext.getContentResolver(), inImage, "Title", null);
        return Uri.parse(path);
    }

    public String getRealPathFromURI(Uri uri) {
        String path = "";
        if (getContentResolver() != null) {
            Cursor cursor = getContentResolver().query(uri, null, null, null, null);
            if (cursor != null) {
                cursor.moveToFirst();
                int idx = cursor.getColumnIndex(MediaStore.Images.ImageColumns.DATA);
                path = cursor.getString(idx);
                cursor.close();
            }
        }
        return path;
    }


    private void requeststorage() {


        if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE)) {

            new AlertDialog.Builder(this)
                    .setTitle("درخواست مجوز")
                    .setMessage("برای دسترسی به حافظه باید مجوز را تایید کنید")
                    .setPositiveButton("موافقم", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {

                            reqPermission();

                        }


                    })
                    .setNegativeButton("لغو", new DialogInterface.OnClickListener() {
                        @Override
                        public void onClick(DialogInterface dialogInterface, int i) {

                            dialogInterface.dismiss();

                        }
                    })
                    .create()
                    .show();

        } else {

            reqPermission();

        }
    }

    private void reqPermission() {

        ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, CAMERA_REQUEST_CODE);

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_REQUEST_CODE) {

            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                Toast.makeText(this, "مجوز تایید شد", Toast.LENGTH_SHORT).show();

                File imgFile = new  File(String.valueOf(finalFile));
                if(imgFile.exists()){
                  //  Toast.makeText(this, "okkkkkkkkkkkkkkkkkkkkkkk", Toast.LENGTH_SHORT).show();
                  //  uploadFoto();
                }
                else
                {     //       Toast.makeText(this, "no kay", Toast.LENGTH_SHORT).show();

                }


            } else {

                Toast.makeText(this, "مجوز رد شد", Toast.LENGTH_SHORT).show();

            }

        }
    }

    ///
    private void uploadFoto() {
        String image_path = "/storage/emulated/0/Pictures/1613250115220.jpg";
        File imageFile = new File(image_path);

        OkHttpClient client = new OkHttpClient().newBuilder().readTimeout(300000, TimeUnit.MILLISECONDS).build();

        MediaType mediaType = MediaType.parse("text/plain");
        RequestBody body = new MultipartBody.Builder().setType(MultipartBody.FORM).addFormDataPart("docfile", String.valueOf(finalFile), RequestBody.create(MediaType.parse("application/octet-stream"), new File(String.valueOf(finalFile)))).build();
        Request request = new Request.Builder().url("http://af510a271628.ngrok.io").method("POST", body).build();
//        final OkHttpClient okHttpClient = new OkHttpClient.Builder()
//                .connectTimeout(20, TimeUnit.SECONDS)
//                .writeTimeout(20, TimeUnit.SECONDS)
//                .readTimeout(30, TimeUnit.SECONDS)
//                .build();
//        new Retrofit.Builder()
//                .client(okHttpClient);

        client.newCall(request).enqueue(new Callback() {

            @Override
            public void onFailure(Call call, IOException e) {
                e.printStackTrace();
            }

            @Override
            public void onResponse(Call call, final Response response) throws IOException {

                if (response.isSuccessful()) {
                    String json= String.valueOf(response.body().string());

                    runOnUiThread(new Runnable() {

                        @Override
                        public void run() {

                          //  String message = "The quick brown fox jumps over the lazy dog.";
                            String str =json;
                            str = str.substring(2, str.length() - 2);
                            str = str.replaceAll("'","");
                            String[] substrings = str.split(",", 2);
                            txtview.setText(null);
                            for (String s : substrings)
                            {
                               // System.out.println(s);
                                txtview.setText(txtview.getText()+"\n"+s);
                               // System.out.println(s);
                            }

                            speak();
                        }
                    });



                   throw new IOException("Unexpected code "+response);

                } else {
                    // todo: inja object response hamoon javabiye ke server bet dade
                }

            }
        });
    }
}




