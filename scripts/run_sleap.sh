## cmd to convert sleap outcome to mp4 with specific marker_size and color
sleap-render labels_20manual_trained_infer80Frame_manual80Correction_trainedModel_test_v1.v010.slp.nwb.slp  -o out_test4.mp4 -f 30 --marker_size 3 --palette solarized

sleap-render labels_153labels_upbodyCenter_added_predictionCleaned_trainedModel_170labels.v020.slp -f 30 --marker_size 3 --palette solarized -o out_multiAnimal_11keypoints_1.mp4


## sleap render multiple videos
## there is a weired thing, I have to close the SLEAP first and run the sleap-render
sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_wt_A1.mp4 --fps 30 --frames 0-5000 --video-index 0 --marker_size 3 --palette solarized

sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_wt_A2.mp4 --fps 30 --frames 0-5000 --video-index 1 --marker_size 3 --palette solarized

sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_wt_A3.mp4 --fps 30 --frames 0-5000 --video-index 2 --marker_size 3 --palette solarized

sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_mutant_A1.mp4 --fps 30 --frames 0-5000 --video-index 3 --marker_size 3 --palette solarized

sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_mutant_A2.mp4 --fps 30 --frames 0-5000 --video-index 4 --marker_size 3 --palette solarized

sleap-render allAnimals_11keypoints_625labels_final.v035.slp -o out_multiAnimal_11keypoints_mutant_A3.mp4 --fps 30 --frames 0-5000 --video-index 5 --marker_size 3 --palette solarized

