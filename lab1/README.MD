1. Làm quen môi trường CartPole, MountainCar.
- CartPole-v1:
    - Observation có 4 chiều.
        1. Cart position (vị trí xe)
        2. Cart velocity (vận tốc xe)
        3. Pole angle (góc của cây gậy)
        4. Pole angular velocity (vận tốc góc của cây gậy)
    - State space là liên tục (continuous).
        Vì các giá trị trong observation space là float trong một khoảng liên tục: Box([-4.8, -inf, -0.418, -inf], [4.8, inf, 0.418, inf])
    - Action space là rời rạc (discrete).
        Có 2 hành động:
            0 → đẩy xe sang trái
            1 → đẩy xe sang phải
    - Reward được tính:
        Reward = +1 cho mỗi bước thời gian nếu pole chưa bị ngã.
        Mục tiêu của agent:
            giữ cây gậy cân bằng càng lâu càng tốt.
    - Episode kết thúc khi:
        1. Góc của pole vượt quá khoảng cho phép (~12°)
        2. Xe di chuyển ra ngoài phạm vi cho phép (~±2.4)
        3. Hoặc đạt số bước tối đa của environment
- MountainCar-v0:
    - Observation có 2 chiều.
        1. Car position (vị trí xe)
        2. Car velocity (vận tốc xe)
    - State space là liên tục (continuous).
        Observation space:  Box([-1.2, -0.07], [0.6, 0.07]). Giá trị là float trong một khoảng liên tục.
    - Action space là rời rạc (discrete).
        Có 3 hành động:
            0 → đẩy xe sang trái
            1 → không đẩy
            2 → đẩy xe sang phải
    - Reward được tính:
        Reward = -1 cho mỗi bước thời gian.
        Điều này khuyến khích agent: lên tới đỉnh núi nhanh nhất có thể.
    - Episode kết thúc khi:
        1. Xe đạt vị trí mục tiêu (position ≥ 0.5)
        2. Hoặc đạt số bước tối đa của environment (200 bước)

2. Tạo Random Agent & Quy trình RL sử dụng 2 môi trường CartPole và MountainCar 
    - Reward trung bình của random agent là bao nhiêu:
        CartPole-v1: ~21.47
        MountainCar-v0: -200.0
        -> Giải thích: CartPole có reward dương, dao động nhiều vì Random Agent đôi khi giữ được thăng bằng lâu. MountainCar luôn -200 vì Random Agent không học được cách leo dốc, nên kết thúc mỗi episode với reward cực thấp.
    
    -  Có xu hướng tăng theo thời gian không
        CartPole-v1: Không có xu hướng tăng ổn định. Reward lên xuống thất thường, chỉ một số episode “may mắn” được reward cao, dao động mức trung bình.
        MountainCar-v0: Hoàn toàn không tăng, luôn giữ giá trị -200, do hành động random không đủ để lên đỉnh.
        -> Giải thích: Random Agent không có policy học được, nên không cải thiện theo thời gian
    
    - Random agent có hội tụ không: Không
        CartPole dao động mạnh, MountainCar luôn -200 
        → Random Agent không học từ trải nghiệm, nên không hội tụ đến giá trị reward tốt hơn.
    
    - Vì sao cần một policy học được:
        Random Agent không có trí tuệ, chỉ chọn action ngẫu nhiên → hiệu suất thấp.
        Một policy học được (ví dụ Q-learning, DQN, hoặc policy gradient) sẽ học cách chọn action tốt dựa trên trạng thái, tối ưu reward.
        Policy học được giúp agent hội tụ, cải thiện reward theo thời gian và đạt hiệu suất ổn định.

3. Mô hình hóa CartPole dưới dạng MDP 
    - Một MDP = (S, A, P, R, γ):
        - State (S)
            - CartPole có 4 biến liên tục:
                s = (x, x_dot, theta, theta_dot)

                x         : vị trí xe
                x_dot     : vận tốc xe
                theta     : góc của pole
                theta_dot : vận tốc góc
        -> State space continuous
        - Action (A)
            A = {0, 1}

            0 : đẩy xe sang trái
            1 : đẩy xe sang phải
        -> Action space discrete () 
        - Reward (R)
            R = +1 cho mỗi bước nếu chưa thất bại
        -> Episode càng lâu → reward càng lớn.
        - Terminal condition
            Episode kết thúc khi:
            |theta| > 12° 
            hoặc
            |x| > 2.4
            hoặc
            step > 500