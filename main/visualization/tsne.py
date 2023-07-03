from yellowbrick.text import TSNEVisualizer

from decorator import timer_decorator


@timer_decorator
def draw_TSNE(X, y, title=None, labels=None, colors=None):
    tsne = TSNEVisualizer(title=title)

    if labels is not None:
        tsne.labels = labels

    if colors is not None:
        tsne.color = colors

    tsne.fit(X, y)
    tsne.show()
