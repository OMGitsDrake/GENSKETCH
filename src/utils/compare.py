import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, help='Percorso al file JSON da analizzare')
    parser.add_argument('--path-cmp', type=Path, help='Percorso al secondo file JSON da analizzare')
    args = parser.parse_args()
    
    with open(args.path, 'r', encoding='utf-8') as f:
        json_obj = json.load(f)
        r_list = json_obj['fold_results']

        if args.path_cmp is not None:
                with open(args.path_cmp, 'r', encoding='utf-8') as f_cmp:
                    json_obj_cmp = json.load(f_cmp)
                    mean_test_loss_1 = json_obj['mean_test_loss']
                    mean_test_loss_2 = json_obj_cmp['mean_test_loss']
                    mean_test_accuracy_1 = json_obj['mean_test_accuracy']
                    mean_test_accuracy_2 = json_obj_cmp['mean_test_accuracy']

                    cmp_out = {
                        'mean_test_loss_diff': mean_test_loss_1 - mean_test_loss_2,
                        'mean_test_accuracy_diff': mean_test_accuracy_1 - mean_test_accuracy_2,
                    }

                    print(json.dumps(cmp_out, indent=2))

        best_fold = None
        best_loss = None
        mean_test_loss = 0
        mean_test_accuracy = 0
        for result in r_list:
            mean_test_loss += result['test_loss']
            mean_test_accuracy += result['test_acc']

            if best_loss is None or result['val_loss'] < best_loss:
                best_loss = result['val_loss']
                best_fold = result['fold']

        out = {
            'mean_test_loss': mean_test_loss/len(r_list),
            'mean_test_accuracy': mean_test_accuracy/len(r_list),
            'best_fold': best_fold,
            'best_loss': best_loss,
        }

        print(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()