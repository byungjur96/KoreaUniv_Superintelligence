# Compute FID, IS
from pytorch_gan_metrics import (
    get_inception_score_from_directory,
    get_fid_from_directory,
    get_inception_score_and_fid_from_directory)

# 

(IS, IS_std), FID = get_inception_score_and_fid_from_directory(
    # '/workspace/results_image/results/Batch1/', 
    # '/workspace/results_image/results/Batch4/', 
    # '/workspace/results_image/results/RadBERT/', 
    # '/workspace/results_image/results/SapBERT/', 
    # '/workspace/results_image/results/zero_init/', 
    # '/workspace/results_image/results/zero_init2/', 
    '/workspace/results_image/results/Vanila/', 
    # '/workspace/results_image/results/ori/', 
    '/workspace/generative_score_metrics/npy_files/test_stats.npz', use_torch=False, batch_size=300)

# Batch1 / Batch4 / RadBERT
# sapbert / zeroinit / zeroinit2
print('IS:', IS)
print('IS_std:', IS_std)
print('FID:', FID)
