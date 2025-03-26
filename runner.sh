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

trainipose() {
    CUDA_VISIBLE_DEVICES=2 python main_implicit_pose.py \
    --config human36m_ipose.yml \
    --doc "ipose_nowarmstart" \
    --use_implicit \
    --train \
    --actions "*" \
    --track_metrics \
    --implicit_iters 20 \
    --implicit_tol 1e-5 \
    --min_iterations 10 \
    --use_memory_efficient
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
    trainipose)
        trainipose
        ;;
    testcpn)
        testcpn
        ;;
    testgt)
        testgt
        ;;
    *)
        echo "Usage: $0 {traincpn|traingt|trainipose|testcpn|testgt}"
        exit 1
esac
exit 0
