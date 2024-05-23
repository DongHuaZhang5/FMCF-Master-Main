from solidity_parser import parser
import json

def transform_ast(ast):
    transformed_nodes = []

    def traverse(node):
        transformed_node = {
            'id': node.get('id'),
            'type': node.get('type'),
            'value': node.get('value'),
            'children': [],
        }

        for child_id in node.get('children', []):
            child_node = next((n for n in ast['nodes'] if n.get('id') == child_id), None)
            if child_node:
                transformed_node['children'].append(traverse(child_node))

        return transformed_node

    for node in ast.get('nodes', []):
        if node.get('type') == 'SourceUnit':
            for child_id in node.get('children', []):
                child_node = next((n for n in ast['nodes'] if n.get('id') == child_id), None)
                if child_node and child_node.get('type') == 'ContractDefinition':
                    transformed_nodes.append(traverse(child_node))
            break

    return transformed_nodes

contract_file = '1_0.txt'
output_file = 'ast.json'

with open(contract_file, 'r') as file:
    contract_code = file.read()

ast = parser.parse(contract_code)
transformed_ast = transform_ast(ast)

with open(output_file, 'w') as file:
    json.dump(transformed_ast, file, indent=2)

print('AST saved to', output_file)
