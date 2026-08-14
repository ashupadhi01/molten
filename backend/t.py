import asyncio
import time

def test_1():
    for _ in range(5):
        print("YO!")
        time.sleep(1)
        # await asyncio.sleep(2)

def test_2():
    for _ in range(5):
        print("MAN")
        time.sleep(0.5)
        # await asyncio.sleep(0.5)

async def root():
    asyncio.create_task(test_1)
    asyncio.create_task(test_2)
    # res1 = await asyncio.to_thread(test_1)
    # print("RES 1: ", res1)
    # res2 = await asyncio.to_thread(test_2)
    # print("RES 2: ", res2)
    

asyncio.run(root())

