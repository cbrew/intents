import os
import re
import pandas as pd

if __name__ == "__main__":
    extractor = re.compile(r"report_k(?P<k>\d).csv")
    root_extractor = re.compile(r"reports/(?P<dataset>[^/]+)/(?P<model>.*)/\d.*")
    res = []
    for root,dirs,files in os.walk("reports"):
        for file in files:
            m1 = extractor.match(file)
            if m1:
                m2 = root_extractor.match(root)
                if m1 and m2:
                    df = pd.read_csv(os.path.join(root,file))
                    r = dict(df.iloc[-2])
                    r['k'] = m1.group("k")
                    r['dataset'] = m2.group("dataset")
                    r['model'] = m2.group("model")
                    res.append(r)
    df_res = pd.DataFrame(res)
    print(df_res.to_string())