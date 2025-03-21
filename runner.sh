traincpn() {
    CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py --train \
    --config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_cpn.pth \
    --doc human36m_diffpose_uvxyz_cpn --exp exp --ni \
    >exp/human36m_diffpose_uvxyz_cpn.out 2>&1 &
}

traingt() {
    python main_diffpose_frame.py --train \
    --config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --doc human36m_diffpose_uvxyz_gt --exp exp --ni \
    >exp/human36m_diffpose_uvxyz_gt.out 2>&1 &
}

trainimp() {
    python main_implicit_diffpose_frame.py \
    --config human36m_diffpose_uvxyz_implicit.yml \
    --model_type implicit \
    --train \
    --doc implicit_model_run1 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth
}

trainai() {
    python main_ai_pose.py \
    --config human36m_aipose_uvxyz.yml \
    --use_adaptive --train \
    --doc ai_run1 --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --track_metrics --downsample 16
}

traingtcpu() {
    CUDA_VISIBLE_DEVICES="" python main_diffpose_frame.py --train \
    --config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
    --model_pose_path checkpoints/gcn_xyz_gt.pth \
    --doc human36m_diffpose_uvxyz_gt --exp exp --ni \
    >exp/human36m_diffpose_uvxyz_gt.out 2>&1 &
}

testcpn() {
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_cpn.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_cpn.pth \
--model_diff_path checkpoints/diffpose_uvxyz_cpn.pth \
--doc t_human36m_diffpose_uvxyz_cpn --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_cpn.out 2>&1 &
}

testgt() {
CUDA_VISIBLE_DEVICES=0 python main_diffpose_frame.py \
--config human36m_diffpose_uvxyz_gt.yml --batch_size 1024 \
--model_pose_path checkpoints/gcn_xyz_gt.pth \
--model_diff_path checkpoints/diffpose_uvxyz_gt.pth \
--doc t_human36m_diffpose_uvxyz_gt --exp exp --ni \
>exp/t_human36m_diffpose_uvxyz_gt.out 2>&1 &
}

# Main script
case "$1" in
    traincpn)
        traincpn
        ;;
    traingt)
        traingt
        ;;
    trainimp)
        trainimp
        ;;
    traingtcpu)
        traingtcpu
        ;;
    testcpn)
        testcpn
        ;;
    testgt)
        testgt
        ;;
    *)
        echo "Usage: $0 {traincpn|traingt|trainimp|traingtcpu|testcpn|testgt}"
        exit 1
esac
exit 0
