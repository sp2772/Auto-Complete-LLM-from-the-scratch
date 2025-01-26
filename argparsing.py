import argparse
parser=argparse.ArgumentParser(description="Command-line argument parser for llms")

def parse_args():
    
    parser.add_argument('-llms',type = str, required= True,help='Please proved an llm')
    
    return parser.parse_args()

def main():
    args= parse_args()
    print(f'Provided llm is: {args.llms}')
    
if __name__ == '__main__':
    main()

