def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    import matplotlib.pyplot as plt

    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    rendered_count = 0
    for i, (ax, img) in enumerate(zip(axes, imgs, strict=False)):
        rendered_count = i + 1
        try:
            img = img.detach().numpy()
        except Exception:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    for ax in axes[rendered_count:]:
        ax.set_visible(False)
    return axes
