import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.relu(out)

class GomokuNet(nn.Module):
    def __init__(self, board_size=15, input_channels=3, num_res_blocks=5, num_filters=64):
        """
        board_size: the length of one side of the board (assume a square board, e.g., 15x15)
        input_channels: number of channels for the board representation (e.g., your stones, opponent stones, turn indicator)
        num_res_blocks: number of residual blocks to use.
        num_filters: number of filters for convolutional layers.
        """
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.num_moves = board_size * board_size

        # Shared initial convolutional layer.
        self.conv = nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks.
        self.res_layers = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head.
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.num_moves)

        # Value head.
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Shared layers.
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_layers(x)

        # Policy head.
        p = self.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)
        # Use log_softmax so you can interpret outputs as log probabilities.
        policy_output = F.log_softmax(p, dim=1)

        # Value head.
        v = self.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value_output = torch.tanh(v)  # Squashes output to (-1, 1)

        return policy_output, value_output

if __name__ == '__main__':
    # Test the network with a dummy input.
    dummy_input = torch.randn(1, 3, 15, 15)
    model = GomokuNet(board_size=15, input_channels=3, num_res_blocks=5, num_filters=64)
    policy, value = model(dummy_input)
    print("Policy output shape:", policy.shape)  # Expected shape: [1, 225] for a 15x15 board.
    print("Value output:", value)
