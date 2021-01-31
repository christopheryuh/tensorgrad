---
id: doc1
title: Introduction
sidebar_label: Introduction / Getting Started
slug: /
---


# Tensorgrad


Tensorgrad is an easy to use automatic differention engine, built on top of Numpy, and made for deep learning. Tensorgrad is made from the example of, and tested with Pytorch, and features a Pytorch like neural network library.

## Your first neural network

Tensorgrad has a pretty easy neural network library, and thus, makes neural networks every easy to build. This simple neural network only comprises of  

### Making the Dataset
```python
from tensorgrad.engine import Tensor

x = Tensor([[1,0,1],[0,1,0],[1,1,1],[0,0,0],[1,1,0],[0,0,1]])

y = Tensor([1,0,1,0,1,0])

```

Here we have a simple binary classification dataset where the output is just the 1st element of the input.


### Making The Model
```python
from tensorgrad import nn
from tensorgrad.models import Model

model = Model([
    nn.Linear(3,9),
    nn.Relu(),
    nn.Linear(9,6),
    nn.Relu(),
    nn.Linear(6,1),
    nn.Sigmoid()

])
```

Here we make the neural network out of 3 linear layers, Relu Activations for the hidden layers, and Sigmoid on the output.

:::note

On layers that have customizable input and output sizes, you will need to pass both the input and outputs. Nothing is inferred

:::


### Training the Model
```python
from tensorgrad.optimizers import SGD

optimizer = SGD(model.parameters(),lr=0.1)


model.train(x, y, loss_fn=nn.Crossentropy(), epochs=10,label_depth=None,batch_size=1,optimizer=optimizer)
```

Here we have a SGD optimizer with a learning rate of 1 set on the model's perameters. 


# H1 - Create the best documentation

## H2 - Create the best documentation

### H3 - Create the best documentation

#### H4 - Create the best documentation

##### H5 - Create the best documentation

###### H6 - Create the best documentation

---

## Emphasis

Emphasis, aka italics, with *asterisks* or _underscores_.

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

---

## Lists

1. First ordered list item
1. Another item
   - Unordered sub-list.
1. Actual numbers don't matter, just that it's a number
   1. Ordered sub-list
1. And another item.

* Unordered list can use asterisks

- Or minuses

+ Or pluses

---

## Links

[I'm an inline-style link](https://www.google.com/)

[I'm an inline-style link with title](https://www.google.com/ "Google's Homepage")

[I'm a reference-style link][arbitrary case-insensitive reference text]

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. http://www.example.com/ or <http://www.example.com/> and sometimes example.com (but not on GitHub, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org/
[1]: http://slashdot.org/
[link text itself]: http://www.reddit.com/

---

## Images

Here's our logo (hover to see the title text):

Inline-style: ![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png 'Logo Title Text 1')

Reference-style: ![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png 'Logo Title Text 2'

Images from any folder can be used by providing path to file. Path should be relative to markdown file.

![img](../static/img/logo.svg)

---

## Code

```javascript
var s = 'JavaScript syntax highlighting';
alert(s);
```

```python
s = "Python syntax highlighting"
print(s)
```

```
No language indicated, so no syntax highlighting.
But let's throw in a <b>tag</b>.
```

```js {2}
function highlightMe() {
  console.log('This line can be highlighted!');
}
```

---

## Tables

Colons can be used to align columns.

| Tables        |      Are      |   Cool |
| ------------- | :-----------: | -----: |
| col 3 is      | right-aligned | \$1600 |
| col 2 is      |   centered    |   \$12 |
| zebra stripes |   are neat    |    \$1 |

There must be at least 3 dashes separating each header cell. The outer pipes (|) are optional, and you don't need to make the raw Markdown line up prettily. You can also use inline Markdown.

| Markdown | Less      | Pretty     |
| -------- | --------- | ---------- |
| _Still_  | `renders` | **nicely** |
| 1        | 2         | 3          |

---

## Blockquotes

> Blockquotes are very handy in email to emulate reply text. This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can _put_ **Markdown** into a blockquote.

---

## Inline HTML

<dl>
  <dt>Definition list</dt>
  <dd>Is something people use sometimes.</dd>

  <dt>Markdown in HTML</dt>
  <dd>Does *not* work **very** well. Use HTML <em>tags</em>.</dd>
</dl>

---

## Line Breaks

Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a _separate paragraph_.

This line is also a separate paragraph, but... This line is only separated by a single newline, so it's a separate line in the _same paragraph_.

---

## Admonitions

:::note

This is a note

:::

:::tip

This is a tip

:::

:::important

This is important

:::

:::caution

This is a caution

:::

:::warning

This is a warning

:::
