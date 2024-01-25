│
├── Cgan_experiment                       <- CGAN models for all epochs and training statistics
│
├── gan_experiment                        <- GAN models for all epochs and training statistics
│
├── gan_interview                         <- Main Code  
│   ├── arg_extractor.py                  <- Extracts Terminal Arguments
│   ├── create_grid_images.py             <- Create Grid of Images Seen in Report
│   ├── experiment_builder.py             <- Main Training Logic to Train an Experiment 
│   ├── get_gan_output.py                 <- Get a Single Output Image from a trained model
│   ├── model_architectures.py            <- GAN's models
│   ├── plot_curves.ipynb                 <- Plot Training Curves
│   ├── storage_utils.py                  <- Helper to Store Stats.
│   └── train_evaluate_gan.py             <-Helper before main logic in experiment_builder.py
│
├── images_for_report   
│   
├── run_cgan.sh                           <- File to run CGAN
├── run_gan.sh                            <- File to run GAN
│ 
└── README.md                             <- The document you are currently reading, written for developers to replicate 
                                             the environment used in the research project
