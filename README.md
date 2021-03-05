This is an implement of our pre-print paper:

Yudi Dong and Huaxia Wang and Yu-Dong Yao, “A Robust Adversarial Network-Based End-to-End Communications System With Strong Generalization Ability Against Adversarial Attacks”, https://arxiv.org/abs/2103.02654


# TensorFlow Version
Our codes are based on TensorFlow-GPU 2.0


# Main Function Files:
"gan_blackbox.py": BLER Peformance of our proposed method under black-box attacks

"gan_whitebox.py": BLER Peformance of our proposed method under white-box attacks

"regular_training_blackbox.py": BLER Peformance of regular training method under black-box attacks

"regular_training_whitebox.py": BLER Peformance of regular training method under white-box attacks

"adversarial_training_blackbox.py": BLER Peformance of adversarial training method under black-box attacks

"adversarial_training_whitebox.py": BLER Peformance of adversarial training method under white-box attacks

# Class Function Files:
"classes/GAN_Classes.py": Implement for our proposed GAN-based end-to-end system

"classes/Autoencoder_Classes.py": Implement for the autoencoder end-to-end system

"classeshamming.py": Implement for the traditional communications system (BPSK, Hamming)

# Other Files:
"UAP": perturbations used for black-box attacks

# Coding Reference:
[1] https://github.com/meysamsadeghi/Security-and-Robustness-of-Deep-Learning-in-Wireless-Communication-Systems
