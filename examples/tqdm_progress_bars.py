from tqdm.auto import tqdm
import time

for i in range(5):
    print(i)
    time.sleep(0.5)


for i in tqdm(range(5)):
    print(i)
    time.sleep(0.5)
