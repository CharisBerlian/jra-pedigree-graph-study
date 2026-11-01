"""Exercise the actual amended preamble with a missing scratch parent."""
from pathlib import Path
import ast,tempfile
HERE=Path(__file__).resolve().parent;PRIVATE=HERE/'private_data'


def main():
    original=(PRIVATE/'parent_prepare_v4.py').read_bytes()
    revised=(PRIVATE/'parent_prepare_v4_mkdir2.py').read_bytes()
    assert revised.replace(b'inp.mkdir(parents=True);bundle.mkdir()',b'inp.mkdir();bundle.mkdir()')==original
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)/'absent'/'tmp';assert not root.exists()
        tree=ast.parse(revised.decode('utf-8'));tree.body=tree.body[:7]
        class Redirect(ast.NodeTransformer):
            def visit_Constant(self,node):
                if node.value=='/kaggle/tmp/focused-stage-input':node.value=str(root/'focused-stage-input')
                elif node.value=='/kaggle/tmp/focused-stage-code':node.value=str(root/'focused-stage-code')
                return node
        tree=Redirect().visit(tree);exec(compile(tree,'<synthetic-preamble>','exec'),{})
        assert (root/'focused-stage-input').is_dir() and (root/'focused-stage-code').is_dir()
    print('PASS actual generated preamble creates a missing scratch parent')


if __name__=='__main__':main()
