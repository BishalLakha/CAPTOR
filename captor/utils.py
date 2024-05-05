from tqdm import tqdm
import os

def get_batch(iterable, n=1, desc='Progress'):
    l = len(iterable)
    with tqdm(total=l, desc=desc) as pbar:
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
            pbar.update(n)


def get_frequency(df):
    "Return a static graph along with edge frequency"
    df['frequency'] = df.groupby(['src_computer', 'dst_computer']).transform('count')
    df = df.drop_duplicates(subset=['src_computer', 'dst_computer'])

    return df


# check path and create directory if not exists
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path