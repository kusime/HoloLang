this project is a multi-language TTS project, it can support Chinese, English, Japanese, Korean, and other languages, and it can support multiple TTS engines, such as gptsovits, vits, bark, etc.


but this project structure is a bit messy, and it is not very efficient, so I want to refactor it.

so i want let you know the basic information of this project, and then you can refactor it.


the pipeline is like this:
1. segment the text into multiple segments
2. for each segment, use TTS to generate audio
3. for each segment, use WhisperX to generate character timestamps
4. merge the audio and timestamps into a single output


about the file structure:
./
├── multiple_pipeline_api.py
  expose an api endpoint to make fontend 
./

pipeline_service.py
pure pipeline 

s3_client.py
pure s3 client 

tts_client.py
call the tts engine endpoint


whisperx_single_client.py
whisperx single language client,which can accpet text language,and text ,and audio bytes,and return char timestamps

# refactor goal
1. make the project more efficient
2. make the project more modular
3. make the project more maintainable
4. make the project more scalable
5. make the project more testable
6. make the project more readable
7. make the project more maintainable
8. make the project more scalable
9. make the project more testable
10. make the project more readable

