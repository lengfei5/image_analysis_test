## cmd to convert sleap outcome to mp4 with specific marker_size and color
sleap-render labels_20manual_trained_infer80Frame_manual80Correction_trainedModel_test_v1.v010.slp.nwb.slp  -o out_test4.mp4 -f 30 --marker_size 3 --palette solarized

sleap-render labels_153labels_upbodyCenter_added_predictionCleaned_trainedModel_170labels.v020.slp -f 30 --marker_size 3 --palette solarized -o out_multiAnimal_11keypoints_1.mp4

