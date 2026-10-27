def is_shuangsheng(s):
    if not s:
        return False
    count0 = s.count('0')
    return count0 % 2 == 0 and len(s) % 2 == 0

def solve():
    n = int(input())
    s = input()

    wins = 0

    for i in range(1, n + 1):
        rem_s = s[i:]

        can_win = False
        for j in range(1, len(rem_s) + 1):
            final_s = rem_s[:j]
            if is_shuangsheng(final_s):
                can_win = True
                break

        if can_win:
            wins += 1

    prob = wins / n
    print(f"{prob:.10f}")

solve() 