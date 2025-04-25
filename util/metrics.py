import numpy as np
from skimage.metrics import structural_similarity, mean_squared_error


def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
           f"{str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}")

    assert org_img.shape == pred_img.shape, msg


def normalize_minmse(x, target):
    """Affine rescaling of x, such that the mean squared error to target is minimal."""
    cov = np.cov(x.flatten(), target.flatten())
    alpha = cov[0, 1] / (cov[0, 0] + 1e-10)
    beta = target.mean() - alpha * x.mean()
    return alpha * x + beta


def rmse(org_img: np.ndarray, pred_img: np.ndarray, max_p=4095) -> float:
    """
    Root Mean Squared Error

    Calculated individually for all bands, then averaged
    """
    _assert_image_shapes_equal(org_img, pred_img, "RMSE")

    org_img = org_img.astype(np.float32)

    # if org_img.ndim < 3:
    #     org_img = org_img[..., None]
    n_bands = org_img.shape[2] if org_img.ndim == 3 else 1

    rmse_bands = []
    for i in range(n_bands):
        dif = np.subtract(org_img, pred_img)
        m = np.mean(np.square(dif / max_p))
        s = np.sqrt(m)
        rmse_bands.append(s)

    return np.mean(rmse_bands)


def psnr(org_img: np.ndarray, pred_img: np.ndarray, max_p=4095) -> float:
    """
    Peek Signal to Noise Ratio, implemented as mean squared error converted to dB.

    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)

    When using 12-bit imagery MaxP is 4095, for 8-bit imagery 255. For floating point imagery using values between
    0 and 1 (e.g. unscaled reflectance) the first logarithmic term can be dropped as it becomes 0
    """
    _assert_image_shapes_equal(org_img, pred_img, "PSNR")

    org_img = org_img.astype(np.float32)

    org_img = org_img[..., np.newaxis] if len(org_img.shape) < 3 else org_img
    pred_img = pred_img[..., np.newaxis] if len(pred_img.shape) < 3 else pred_img

    mse_bands = []
    for i in range(org_img.shape[2]):
        mse_bands.append(np.mean(np.square(org_img[:, :, i] - pred_img[:, :, i])))

    return 20 * np.log10(max_p) - 10. * np.log10(np.mean(mse_bands))


def PSNR(gt, pred, max_p=255.0):
    mse = np.mean((gt - pred) ** 2)
    return 20 * np.log10((max_p) / np.sqrt(mse))


def ssim(org_img: np.ndarray, pred_img: np.ndarray, max_p=4095) -> float:
    """
    Structural SIMularity index
    """
    _assert_image_shapes_equal(org_img, pred_img, "SSIM")

    return structural_similarity(org_img, pred_img, data_range=max_p, multichannel=True)


def compute_all_metrics(img_input, img_pred, img_gt, dtype=np.float32):
    img_input = img_input.squeeze().cpu().numpy().astype(dtype)
    img_pred = img_pred.squeeze().cpu().numpy().astype(dtype)
    img_gt = img_gt.squeeze().cpu().numpy().astype(dtype)

    # compute dynamic range
    range_PSNR = np.max(img_gt) - np.min(img_gt)

    # # compute profile, PSNR, SSIM and NRMSE
    # diag_in = np.diag(img_input)
    # diag_gt = np.diag(img_gt)
    # diag_pred = np.diag(img_pred)

    psnr_in = PSNR(img_gt, img_input, max_p=range_PSNR)
    psnr_pred = PSNR(img_gt, img_pred, max_p=range_PSNR)
    ssim_in = ssim(img_gt, img_input, max_p=range_PSNR)
    ssim_pred = ssim(img_gt, img_pred, max_p=range_PSNR)
    nrmse_in = rmse(img_input, normalize_minmse(img_input, img_gt))
    nrmse_pred = rmse(img_pred, normalize_minmse(img_pred, img_gt))

    return {
        'psnr':
            {
                'in': psnr_in,
                'pred': psnr_pred,
            },
        'ssim':
            {
                'in': ssim_in,
                'pred': ssim_pred,
            },
        'nrmse':
            {
                'in': nrmse_in,
                'pred': nrmse_pred,
            },
        # 'diag':
        #     {
        #         'in': diag_in,
        #         'pred': diag_pred,
        #         'gt': diag_gt,
        #     }
    }


def compute_moments(metrics):
    moments = []

    for m_ in metrics:
        moments.append([m_['psnr']['in'], m_['psnr']['pred'],
                        m_['ssim']['in'], m_['ssim']['pred'],
                        m_['nrmse']['in'], m_['nrmse']['pred']])

    moments = np.asarray(moments)
    moments_mean = moments.mean(axis=0)
    moments_std = moments.std(axis=0)
    moments_labels = ['psnr_in', 'psnr_pred', 'ssim_in', 'ssim_pred', 'nrmse_in', 'nrmse_pred']

    return moments_mean, moments_std, moments_labels
