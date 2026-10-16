global asm_func

section .text
asm_func
    shl    r8, 3                     ; elements -> bytes
    vbroadcastsd ymm2, xmm3          ; ymm2 = {k, k, k, k}
    xor    rax, rax

.loop:
    cmp    rax, r8
    jge    .done

    vmovupd ymm0, [rcx + rax]
    vmulpd  ymm0, ymm0, ymm2         ; 4 elements * k at once
    vmovupd [rdx + rax], ymm0

    add    rax, 32
    jmp    .loop

.done:
    vzeroupper
    ret