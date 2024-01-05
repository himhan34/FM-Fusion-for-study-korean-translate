# Statistic summarize the label likelihood matrix

1. Using [RAM-Grounded-SAM](https://github.com/glennliu/Grounded-Segment-Anything) to generate frame-wise detections with segmentation masks.

2. Summarize association result for each scan.
```python
python python/measure_model.py
```
The result should be saved at a ```RESULT_FOLDER```.

3. 
Based on results in ```RESULT_FOLDER```, calculate label likihood matrix. 
```python
python python/analysis.py
```
The final likelihood matrices are saved at ```measurement_model``` of the project.


