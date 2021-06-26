## Execute
1. Run the following command to download pretrained model
    ```bash
    chmod +x download.sh
    ./download.sh
    ```
2. Put ```.jpg``` files (e.g., test.jpg) into the ```image``` folder. 
    - E.g., `image/computer.jpg`
3. Run the following command
    ```bash
    python main.py --config argument.yml --pose 
    ```
    - use --pose to specify the path of camera pose file(default ```./cam_pose.json```)
4. The rendered videos are stored in ```video/```:
    - E.g., `video/computer_novel.mp4`
