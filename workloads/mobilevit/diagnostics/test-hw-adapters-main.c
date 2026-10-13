#include "test_maps_hw_adapters_task_bin.h"
#include "tile.h"
#include "fsync.h"
#include "eventunit.h"
#include "redmule.h"
#include "utils/maps_operations.h"
#include "utils/printf.h"
@DECLARATIONS@
typedef struct {
    const char *kind_name;
    slice_desc_t slices[4];
    op_desc_t op;
    const uint8_t *inputs[2][3];
    uint32_t sizes[2][3];
    const uint8_t *gold[2];
    uint32_t output_bytes;
} test_case_t;
static const test_case_t cases[] = {@CASES@};
static volatile uint32_t errors;
static inline uint32_t clean_half(uint32_t x) {
    __asm__ volatile("slli %0, %0, 16\n\tsrli %0, %0, 16" : "+r"(x));
    return x;
}
int main(void) {
    uint32_t hartid = get_hartid();
    idma_config_t ic = {.hartid=hartid};
    idma_controller_t idma = {.cfg=&ic,.api=&idma_api};
    redmule_config_t rc = {.hartid=hartid};
    redmule_controller_t redmule = {.cfg=&rc,.api=&redmule_api};
    fsync_config_t fc = {.hartid=hartid};
    fsync_controller_t fsync = {.cfg=&fc,.api=&fsync_api};
    eu_config_t ec = {.hartid=hartid};
    eu_controller_t eu = {.cfg=&ec,.api=&eu_api};
    idma_init(&idma); redmule_init(&redmule); fsync_init(&fsync); eu_init(&eu);
    eu_clear_events(0xffffffffu);
    eu_fsync_init(&eu,0u); eu_idma_init(&eu,0u); eu_redmule_init(&eu,0u);
    if (hartid == 0) {
        maps_operation_runtime_t runtime = {
            .idma_ctrl=&idma,.redmule_ctrl=&redmule,.eu_ctrl=&eu,
            .kernel_abi_version=2,.task_bundle_abi_version=2,
            .spatz_binary_start=SPATZ_BINARY_START,
            .spatz_params=(void *)(get_l1_base(hartid)+MAPS_OPERATION_TASK_SCRATCH_OFFSET),
            .spatz_params_bytes=MAPS_OPERATION_TASK_SCRATCH_BYTES,
            .depthwise_conv_fp16_task=DEPTHWISE_CONV_FP16_SPATZ_TASK,
        };
        int init=maps_operation_runtime_init(&runtime);
        errors=init != 0;
        printf("HW_INIT code=%d cases=%u\n",init,(uint32_t)(sizeof(cases)/sizeof(cases[0])));
        for (uint32_t c=0; !init && c<sizeof(cases)/sizeof(cases[0]); ++c) {
            const test_case_t *tc=&cases[c];
            tile_plan_t plan={.hartid=hartid,.l1_data_base=get_l1_base(hartid),.num_slices=4,.slices=tc->slices};
            for (uint32_t token=0; token<2; ++token) {
                for (uint32_t j=0;j<tc->op.num_inputs;++j)
                    errors += maps_idma_copy_1d_ticketed(&idma,0u,(uint32_t)tc->inputs[token][j],plan.l1_data_base+tc->slices[j].l1_offset_bytes,tc->sizes[token][j],&eu) != 0;
                volatile uint16_t *actual=(volatile uint16_t *)(plan.l1_data_base+tc->slices[3].l1_offset_bytes);
                for (uint32_t i=0;i<tc->output_bytes/2;++i) actual[i]=0;
                int result=maps_execute_operation(&plan,&tc->op,0,&runtime);
                uint32_t bad=0;
                const uint16_t *expected=(const uint16_t *)tc->gold[token];
                for (uint32_t i=0;i<tc->output_bytes/2;++i) {
                    uint32_t a=clean_half(actual[i]), b=clean_half(expected[i]);
                    if (a!=b && (tc->op.kind!=OP_DEPTHWISE_CONV || ((a|b)&0x7fff))) {
                        if (bad<4) printf("HW_BAD case=%u token=%u index=%u actual=%04x expected=%04x\n",c,token,i,a,b);
                        ++bad;
                    }
                }
                errors += bad + (result!=0);
                printf("HW_CHECK case=%u token=%u kind=%s values=%u errors=%u code=%d\n",c,token,tc->kind_name,tc->output_bytes/2,bad,result);
            }
        }
        spatz_clk_dis();
    }
    fsync_sync_global(&fsync); eu_fsync_wait(&eu,WFE);
    return errors ? -1 : 0;
}
