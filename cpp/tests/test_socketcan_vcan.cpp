#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "bev/radarcam/socketcan_reader.hpp"
#include "fixtures/can_decoder_golden.hpp"

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

using bev::radarcam::SocketCanReader;
namespace golden = bev::radarcam::test_fixtures;

namespace {

// Opens a raw CAN-FD TX socket bound to `iface`; returns -1 on any failure (interface missing, no
// permission, etc.) rather than throwing, so the caller can skip the test gracefully.
int open_tx_socket(const std::string& iface) {
    const int fd = socket(AF_CAN, SOCK_RAW, CAN_RAW);
    if (fd < 0) return -1;

    constexpr int kEnableFd = 1;
    if (setsockopt(fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &kEnableFd, sizeof(kEnableFd)) < 0) {
        ::close(fd);
        return -1;
    }

    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ - 1);
    if (ioctl(fd, SIOCGIFINDEX, &ifr) < 0) {
        ::close(fd);
        return -1;
    }

    struct sockaddr_can addr {};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(fd);
        return -1;
    }
    return fd;
}

bool send_frame(int fd, canid_t can_id, const std::vector<uint8_t>& data) {
    struct canfd_frame frame {};
    frame.can_id = can_id;
    frame.len = static_cast<uint8_t>(data.size());
    frame.flags = CANFD_BRS;
    std::copy(data.begin(), data.end(), frame.data);
    const ssize_t n = write(fd, &frame, sizeof(frame));
    return n == static_cast<ssize_t>(sizeof(frame));
}

}  // namespace

TEST_CASE("SocketCanReader decodes a live dwell over a real AF_CAN socket (vcan0)",
    "[phase4][socketcan][requires_vcan0]") {
    SocketCanReader reader;
    std::string error;
    if (!reader.start("vcan0", error)) {
        const std::string msg = "vcan0 not available (" + error +
            "). One-time setup: sudo modprobe vcan && sudo ip link add dev vcan0 type vcan && "
            "sudo ip link set up vcan0, then re-run this test.";
        SKIP(msg);
    }

    const int tx_fd = open_tx_socket("vcan0");
    REQUIRE(tx_fd >= 0);

    REQUIRE(send_frame(tx_fd, 0x281, golden::kHeaderDetNumber5));

    std::vector<uint8_t> body1;
    body1.insert(body1.end(), golden::kSlotV1TypicalStationary.begin(), golden::kSlotV1TypicalStationary.end());
    body1.insert(body1.end(), golden::kSlotV2SignBitSet.begin(), golden::kSlotV2SignBitSet.end());
    body1.insert(body1.end(), golden::kSlotV3BoundaryExtremes.begin(), golden::kSlotV3BoundaryExtremes.end());
    REQUIRE(send_frame(tx_fd, 0x282, body1));

    std::vector<uint8_t> body2;
    body2.insert(body2.end(), golden::kSlotV1TypicalStationary.begin(), golden::kSlotV1TypicalStationary.end());
    body2.insert(body2.end(), golden::kSlotV2SignBitSet.begin(), golden::kSlotV2SignBitSet.end());
    REQUIRE(send_frame(tx_fd, 0x283, body2));

    // Poll for the reader's background thread to have processed and published the dwell (up to 1s).
    bool got_it = false;
    for (int i = 0; i < 50 && !got_it; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (reader.snapshot().detections.size() == 5) got_it = true;
    }
    REQUIRE(got_it);

    const auto snap = reader.snapshot();
    CHECK(snap.frame_id == 7);
    CHECK(snap.radar_id == 2);
    REQUIRE(snap.detections.size() == 5);
    CHECK(snap.detections[0].range_m == Catch::Approx(5.0));
    CHECK(snap.detections[1].range_m == Catch::Approx(12.34));

    reader.stop();
    ::close(tx_fd);
}
